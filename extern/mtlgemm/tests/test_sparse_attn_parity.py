"""Parity: fused sparse attention vs SDPA-padded reference on MPS.

The SDPA reference pads variable-length batched sequences into
[N, max_len, H, C], builds a [N, 1, max_q, max_kv] boolean mask, then
calls torch.nn.functional.scaled_dot_product_attention. Our kernel does
the same math on the packed [T, H, C] layout with per-sequence boundary
checks via cu_seqlens — no padding, no mask allocation.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import math
import pytest
import torch
import torch.nn.functional as F

import flex_gemm


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="sparse attention test requires MPS (production target).",
)


def sdpa_padded_reference(q, k, v, q_seqlens, kv_seqlens):
    """Reference implementation of the padded-SDPA path.

    Computed on CPU — this PyTorch build has a broken new_zeros fp32 MPS
    kernel (DispatchStub missing), which the trellis2 production path on
    MPS also has to work around. We compare MPS kernel output against a
    CPU reference instead of an MPS-SDPA reference; the fused kernel is
    still what we're gating on. CPU-run SDPA is a faithful reference for
    the math regardless of how trellis2 dispatches it at runtime.
    """
    q_c = q.detach().cpu().float()
    k_c = k.detach().cpu().float()
    v_c = v.detach().cpu().float()
    dtype_out = q.dtype
    N = len(q_seqlens)
    max_q = max(q_seqlens)
    max_kv = max(kv_seqlens)
    H = q_c.shape[1]
    C_q = q_c.shape[2]
    C_v = v_c.shape[2]

    q_dense = torch.zeros(N, max_q, H, C_q)
    k_dense = torch.zeros(N, max_kv, H, C_q)
    v_dense = torch.zeros(N, max_kv, H, C_v)
    mask = torch.zeros(N, max_q, max_kv, dtype=torch.bool)

    q_off = 0
    kv_off = 0
    for i in range(N):
        ql, kvl = q_seqlens[i], kv_seqlens[i]
        q_dense[i, :ql] = q_c[q_off:q_off + ql]
        k_dense[i, :kvl] = k_c[kv_off:kv_off + kvl]
        v_dense[i, :kvl] = v_c[kv_off:kv_off + kvl]
        mask[i, :ql, :kvl] = True
        q_off += ql
        kv_off += kvl

    q_t = q_dense.permute(0, 2, 1, 3)
    k_t = k_dense.permute(0, 2, 1, 3)
    v_t = v_dense.permute(0, 2, 1, 3)
    float_mask = torch.zeros(N, 1, max_q, max_kv)
    float_mask.masked_fill_(~mask.unsqueeze(1), float('-inf'))

    out_dense = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=float_mask)
    out_dense = out_dense.permute(0, 2, 1, 3)

    out_parts = []
    for i in range(N):
        out_parts.append(out_dense[i, :q_seqlens[i]])
    return torch.cat(out_parts, dim=0).to(dtype_out)


TOL = {
    torch.float32:  dict(atol=5e-4, rtol=1e-4),
    torch.float16:  dict(atol=2e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=5e-2, rtol=2e-2),
}


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlens", [
    [16],                      # N=1, simple
    [16, 16],                  # N=2, equal lengths
    [32, 8, 24],               # N=3, variable
    [4, 64, 16, 32, 48],       # N=5, wide range — stresses the padding waste
])
@pytest.mark.parametrize("H,C_q,C_v", [
    (4, 32, 32),
    (8, 64, 64),
])
def test_sparse_attn_matches_sdpa(seqlens, H, C_q, C_v, dtype):
    torch.manual_seed(0x5AFE ^ sum(seqlens) ^ (H * 31) ^ (C_q * 17))
    device = "mps"

    N = len(seqlens)
    T_q = sum(seqlens)
    T_kv = sum(seqlens)  # self-attention pattern

    # Build on CPU + move — some PyTorch MPS builds have a broken randn fp32
    # MPS kernel (DispatchStub missing).
    q = (torch.randn(T_q, H, C_q, dtype=dtype) * 0.3).to(device)
    k = (torch.randn(T_kv, H, C_q, dtype=dtype) * 0.3).to(device)
    v = (torch.randn(T_kv, H, C_v, dtype=dtype) * 0.3).to(device)
    scale = 1.0 / math.sqrt(C_q)

    # Build cu_seqlens (CPU → MPS, since torch.tensor + cumsum on int may have
    # spotty MPS coverage across builds).
    csq = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlens), 0)]).int().to(device)
    cskv = csq.clone()

    # Our kernel
    out = flex_gemm.kernels.cuda.sparse_attention_fwd(
        q, k, v, csq, cskv, max(seqlens), max(seqlens), scale,
    )
    assert out.device.type == "mps", f"expected MPS output, got {out.device}"
    assert out.shape == (T_q, H, C_v), f"shape {out.shape} != ({T_q},{H},{C_v})"

    # SDPA reference (runs on MPS)
    ref = sdpa_padded_reference(q, k, v, seqlens, seqlens)

    tol = TOL[dtype]
    out_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    diff = (out_cpu - ref_cpu).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = torch.allclose(out_cpu, ref_cpu, **tol)
    assert ok, (
        f"parity failed seqlens={seqlens} H={H} C_q={C_q} C_v={C_v} dtype={dtype}: "
        f"max_err={max_err:.4e} mean_err={mean_err:.4e} "
        f"(atol={tol['atol']}, rtol={tol['rtol']})"
    )
