"""Parity: fused sparse attention backward vs autograd-through-SDPA reference.

Compares our custom bwd kernel's (dQ, dK, dV) against those produced by
feeding the same forward through torch.nn.functional.scaled_dot_product_attention
with padding-mask and calling autograd. The reference runs on CPU because
this PyTorch build has broken MPS kernels for new_zeros fp16/fp32 (same
workaround as the forward parity tests).
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
    reason="sparse attention backward test requires MPS (production target).",
)


def manual_reference_bwd(q, k, v, d_out, q_seqlens, kv_seqlens, scale):
    """Explicit CPU-fp32 reference for attention backward.

    Uses the standard two-pass formula:
       P  = softmax(Q K^T * scale)
       dV = P^T @ dO
       dP = dO @ V^T
       D  = rowsum(P * dP)
       dS = P * (dP - D)
       dQ = dS @ K * scale
       dK = dS^T @ Q * scale

    Avoids autograd because this PyTorch build raises
    "Cannot have both MPS and cuda" on .backward() regardless of tensor
    device — same issue the forward-parity test worked around by running
    SDPA on CPU for math only.
    """
    q_c = q.detach().cpu().float()
    k_c = k.detach().cpu().float()
    v_c = v.detach().cpu().float()
    do_c = d_out.detach().cpu().float()

    dq = torch.zeros_like(q_c)
    dk = torch.zeros_like(k_c)
    dv = torch.zeros_like(v_c)

    H = q_c.shape[1]
    q_off = kv_off = 0
    for i in range(len(q_seqlens)):
        ql = q_seqlens[i]
        kvl = kv_seqlens[i]
        for h in range(H):
            Q  = q_c[q_off:q_off + ql, h, :]       # [ql, C_q]
            K  = k_c[kv_off:kv_off + kvl, h, :]    # [kvl, C_q]
            V  = v_c[kv_off:kv_off + kvl, h, :]    # [kvl, C_v]
            dO = do_c[q_off:q_off + ql, h, :]      # [ql, C_v]

            S = Q @ K.T * scale                    # [ql, kvl]
            P = torch.softmax(S, dim=-1)           # [ql, kvl]

            dV = P.T @ dO                          # [kvl, C_v]
            dP = dO @ V.T                          # [ql, kvl]
            D  = (P * dP).sum(dim=-1, keepdim=True)  # [ql, 1]
            dS = P * (dP - D)                      # [ql, kvl]
            dQ = dS @ K * scale                    # [ql, C_q]
            dK = dS.T @ Q * scale                  # [kvl, C_q]

            dq[q_off:q_off + ql, h, :]   = dQ
            dk[kv_off:kv_off + kvl, h, :] = dK
            dv[kv_off:kv_off + kvl, h, :] = dV
        q_off += ql
        kv_off += kvl
    return dq, dk, dv


TOL = {
    torch.float32:  dict(atol=5e-4, rtol=1e-3),
    torch.float16:  dict(atol=5e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=5e-2),
}


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlens", [
    [16],
    [16, 16],
    [32, 8, 24],
    [4, 48, 16, 32],
])
@pytest.mark.parametrize("H,C_q,C_v", [
    (4, 32, 32),
    (8, 64, 64),
])
def test_sparse_attn_bwd_matches_sdpa(seqlens, H, C_q, C_v, dtype):
    torch.manual_seed(0x8EEF ^ sum(seqlens) ^ (H * 31) ^ (C_q * 19))
    device = "mps"

    T_q = sum(seqlens)
    T_kv = sum(seqlens)

    q = (torch.randn(T_q, H, C_q, dtype=dtype) * 0.3).to(device)
    k = (torch.randn(T_kv, H, C_q, dtype=dtype) * 0.3).to(device)
    v = (torch.randn(T_kv, H, C_v, dtype=dtype) * 0.3).to(device)
    d_out = (torch.randn(T_q, H, C_v, dtype=dtype) * 0.1).to(device)
    scale = 1.0 / math.sqrt(C_q)

    csq = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlens), 0)]).int().to(device)
    cskv = csq.clone()

    dq, dk, dv = flex_gemm.kernels.cuda.sparse_attention_bwd(
        q, k, v, d_out, csq, cskv, max(seqlens), max(seqlens), scale,
    )
    assert dq.device.type == "mps"
    assert dk.device.type == "mps"
    assert dv.device.type == "mps"
    assert dq.shape == (T_q, H, C_q)
    assert dk.shape == (T_kv, H, C_q)
    assert dv.shape == (T_kv, H, C_v)

    dq_ref, dk_ref, dv_ref = manual_reference_bwd(q, k, v, d_out, seqlens, seqlens, scale)

    tol = TOL[dtype]
    for name, ours, ref in [("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)]:
        ours_cpu = ours.detach().cpu().float()
        ref_cpu = ref.detach().cpu().float()
        diff = (ours_cpu - ref_cpu).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        ok = torch.allclose(ours_cpu, ref_cpu, **tol)
        assert ok, (
            f"{name} parity failed seqlens={seqlens} H={H} C_q={C_q} C_v={C_v} "
            f"dtype={dtype}: max_err={max_err:.4e} mean_err={mean_err:.4e} "
            f"(atol={tol['atol']}, rtol={tol['rtol']})"
        )
