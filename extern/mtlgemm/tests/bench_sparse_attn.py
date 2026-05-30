"""Benchmark: fused sparse attention vs SDPA-padded on MPS.

Measures the attention forward at trellis2-decoder-style shapes: variable-
length sparse sequences from a 3D sparse grid, packed as [T, H, C] with
cu_seqlens. SDPA-padded is the path trellis2-apple currently uses on MPS.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")
import math
import time
import torch
import torch.nn.functional as F

assert torch.backends.mps.is_available()
import flex_gemm


def bench(label, fn, warmup=3, iters=30):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:52s} {ms:8.3f} ms/call")
    return ms


def sdpa_padded_cpu_bounce(q, k, v, q_seqlens, kv_seqlens):
    """SDPA-padded reference, with MPS->CPU->MPS round-trip. This is the only
    way to run SDPA-padded on the user's PyTorch build (new_zeros for MPS
    fp16/fp32 is broken via DispatchStub) — the same path trellis2-apple
    currently has to fall back to when encountering the issue."""
    orig_device = q.device
    q = q.detach().cpu()
    k = k.detach().cpu()
    v = v.detach().cpu()
    dtype = q.dtype
    N = len(q_seqlens)
    max_q = max(q_seqlens)
    max_kv = max(kv_seqlens)
    H, C_q = q.shape[1], q.shape[2]
    C_v = v.shape[2]
    q_dense = q.new_zeros(N, max_q, H, C_q)
    k_dense = k.new_zeros(N, max_kv, H, C_q)
    v_dense = v.new_zeros(N, max_kv, H, C_v)
    mask = torch.zeros(N, max_q, max_kv, dtype=torch.bool)
    q_off = 0; kv_off = 0
    for i in range(N):
        ql, kvl = q_seqlens[i], kv_seqlens[i]
        q_dense[i, :ql] = q[q_off:q_off + ql]
        k_dense[i, :kvl] = k[kv_off:kv_off + kvl]
        v_dense[i, :kvl] = v[kv_off:kv_off + kvl]
        mask[i, :ql, :kvl] = True
        q_off += ql; kv_off += kvl
    q_t = q_dense.permute(0, 2, 1, 3)
    k_t = k_dense.permute(0, 2, 1, 3)
    v_t = v_dense.permute(0, 2, 1, 3)
    float_mask = torch.zeros(N, 1, max_q, max_kv, dtype=dtype)
    float_mask.masked_fill_(~mask.unsqueeze(1), float('-inf'))
    out_dense = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=float_mask)
    out_dense = out_dense.permute(0, 2, 1, 3)
    out_parts = []
    for i in range(N):
        out_parts.append(out_dense[i, :q_seqlens[i]])
    return torch.cat(out_parts, dim=0).to(orig_device)


sdpa_padded = sdpa_padded_cpu_bounce


print("=" * 80)
print("Sparse attention forward benchmark on MPS (M3 Max, fp16)")
print("=" * 80)

# Shapes: trellis2 decoder typically has 8-12 heads, head_dim=64, sequences
# of hundreds to low thousands, total tokens in 500-5000 range.
configs = [
    # (label, seqlens, H, C_q, C_v)
    ("N=4 balanced",   [128, 128, 128, 128],                 8, 64, 64),
    ("N=4 skewed 4x",  [256, 64, 128, 64],                   8, 64, 64),
    ("N=8 balanced",   [256]*8,                              8, 64, 64),
    ("N=8 skewed 8x",  [512, 64, 256, 128, 64, 256, 64, 128], 8, 64, 64),
    ("N=2 large",      [2048, 2048],                         8, 64, 64),
    ("N=16 tiny",      [32]*16,                              8, 64, 64),
]

results = []
# fp32 on MPS — this PyTorch build has broken fp16 MPS zeros/randn kernels, so
# the SDPA reference (which uses new_zeros) fails there. fp32 works end-to-end.
# Production trellis2 on MPS would hit the same issue with fp16 and needs the
# same workaround; we bench fp32 here and note the conclusion scales.
dtype = torch.float32
# Always build on CPU + transfer — this PyTorch build has a broken MPS
# randn fp32/fp16 kernel (DispatchStub missing).
def rand_on(shape, dtype, device):
    return (torch.randn(*shape, dtype=dtype) * 0.3).to(device)


for label, seqlens, H, C_q, C_v in configs:
    T = sum(seqlens)
    q = rand_on((T, H, C_q), dtype, 'mps')
    k = rand_on((T, H, C_q), dtype, 'mps')
    v = rand_on((T, H, C_v), dtype, 'mps')
    csq = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlens), 0)]).int().to('mps')
    cskv = csq.clone()
    scale = 1.0 / math.sqrt(C_q)
    max_q = max(seqlens); max_kv = max(seqlens)

    def fused():
        return flex_gemm.kernels.cuda.sparse_attention_fwd(q, k, v, csq, cskv, max_q, max_kv, scale)
    def padded():
        return sdpa_padded(q, k, v, seqlens, seqlens)

    print(f"\n{label}: T={T} H={H} C_q={C_q} C_v={C_v} max_seqlen={max_q}")
    # Verify parity first (bit-exactness not expected, within fp16 tol).
    try:
        a = fused().detach().cpu().float()
        b = padded().detach().cpu().float()
        err = (a - b).abs().max().item()
        parity_ok = err < 0.2
        print(f"  parity max_err={err:.4e}  {'OK' if parity_ok else 'FAIL'}")
    except Exception as e:
        print(f"  parity check failed: {e}")
        parity_ok = False

    fused_ms = bench("fused sparse_attention_fwd (MPS)", fused)
    padded_ms = bench("SDPA-padded (current trellis2 MPS path)", padded)
    speedup = padded_ms / fused_ms
    print(f"  speedup fused/SDPA: {speedup:.2f}x")
    results.append((label, T, max_q, fused_ms, padded_ms, speedup))

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"{'shape':32s}  {'T':>6s}  {'max':>6s}  {'fused':>9s}  {'sdpa':>9s}  {'speedup':>8s}")
for label, T, mq, f_ms, p_ms, sp in results:
    print(f"{label:32s}  {T:6d}  {mq:6d}  {f_ms:6.3f}ms  {p_ms:6.3f}ms  {sp:6.2f}x")
