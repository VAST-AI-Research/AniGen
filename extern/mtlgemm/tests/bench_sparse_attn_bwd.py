"""Benchmark: fused sparse attention bwd on MPS.

Measures the two-pass naive bwd against the bench configs used for the
forward. The reference baseline is an explicit manual CPU bwd (same
formula used in the parity test) rather than autograd-through-SDPA —
this PyTorch build raises "Cannot have both MPS and cuda" when autograd
fires, so we bench the closed-form-math path instead.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")
import math
import time
import torch

assert torch.backends.mps.is_available()
import flex_gemm


def bench(label, fn, warmup=3, iters=20, mps_sync=True):
    for _ in range(warmup):
        fn()
    if mps_sync:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if mps_sync:
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:52s} {ms:8.3f} ms/call")
    return ms


def manual_bwd_cpu(q, k, v, d_out, q_seqlens, kv_seqlens, scale):
    """Explicit closed-form bwd on CPU fp32."""
    q_c  = q.detach().cpu().float()
    k_c  = k.detach().cpu().float()
    v_c  = v.detach().cpu().float()
    do_c = d_out.detach().cpu().float()
    dq = torch.zeros_like(q_c)
    dk = torch.zeros_like(k_c)
    dv = torch.zeros_like(v_c)
    H = q_c.shape[1]
    q_off = kv_off = 0
    for i in range(len(q_seqlens)):
        ql  = q_seqlens[i]
        kvl = kv_seqlens[i]
        for h in range(H):
            Q  = q_c[q_off:q_off + ql, h]
            K  = k_c[kv_off:kv_off + kvl, h]
            V  = v_c[kv_off:kv_off + kvl, h]
            dO = do_c[q_off:q_off + ql, h]
            S = Q @ K.T * scale
            P = torch.softmax(S, dim=-1)
            dV = P.T @ dO
            dP = dO @ V.T
            D  = (P * dP).sum(dim=-1, keepdim=True)
            dS = P * (dP - D)
            dQ = dS @ K * scale
            dK = dS.T @ Q * scale
            dq[q_off:q_off + ql, h]    = dQ
            dk[kv_off:kv_off + kvl, h] = dK
            dv[kv_off:kv_off + kvl, h] = dV
        q_off += ql
        kv_off += kvl
    return dq.to('mps'), dk.to('mps'), dv.to('mps')


print("=" * 80)
print("Sparse attention BACKWARD benchmark on MPS (M3 Max, fp32)")
print("=" * 80)

configs = [
    ("N=4 balanced",   [128, 128, 128, 128],                 8, 64, 64),
    ("N=4 skewed 4x",  [256, 64, 128, 64],                   8, 64, 64),
    ("N=8 balanced",   [256]*8,                              8, 64, 64),
    ("N=8 skewed 8x",  [512, 64, 256, 128, 64, 256, 64, 128], 8, 64, 64),
    ("N=2 large",      [2048, 2048],                         8, 64, 64),
    ("N=16 tiny",      [32]*16,                              8, 64, 64),
]

dtype = torch.float32
def rand_on(shape, d, dev):
    return (torch.randn(*shape, dtype=d) * 0.3).to(dev)

results = []
for label, seqlens, H, C_q, C_v in configs:
    T = sum(seqlens)
    q     = rand_on((T, H, C_q), dtype, 'mps')
    k     = rand_on((T, H, C_q), dtype, 'mps')
    v     = rand_on((T, H, C_v), dtype, 'mps')
    d_out = (torch.randn(T, H, C_v, dtype=dtype) * 0.1).to('mps')
    csq  = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlens), 0)]).int().to('mps')
    cskv = csq.clone()
    scale = 1.0 / math.sqrt(C_q)
    max_q = max(seqlens)

    def fused():
        return flex_gemm.kernels.cuda.sparse_attention_bwd(
            q, k, v, d_out, csq, cskv, max_q, max_q, scale,
        )
    def cpu_manual():
        return manual_bwd_cpu(q, k, v, d_out, seqlens, seqlens, scale)

    print(f"\n{label}: T={T} H={H} C_q={C_q} C_v={C_v} max_seqlen={max_q}")
    # Parity sanity: max err
    try:
        a_dq, a_dk, a_dv = fused()
        b_dq, b_dk, b_dv = cpu_manual()
        err = max((a_dq.cpu().float() - b_dq.cpu().float()).abs().max().item(),
                  (a_dk.cpu().float() - b_dk.cpu().float()).abs().max().item(),
                  (a_dv.cpu().float() - b_dv.cpu().float()).abs().max().item())
        print(f"  parity max_err={err:.4e}  {'OK' if err < 1e-2 else 'FAIL'}")
    except Exception as e:
        print(f"  parity check failed: {e}")

    fused_ms = bench("fused sparse_attention_bwd (MPS)", fused)
    cpu_ms   = bench("manual-math bwd (CPU reference)", cpu_manual, mps_sync=False)
    speedup  = cpu_ms / fused_ms
    print(f"  speedup fused/CPU-manual: {speedup:.2f}x")
    results.append((label, T, max_q, fused_ms, cpu_ms, speedup))

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"{'shape':32s}  {'T':>6s}  {'max':>6s}  {'fused':>9s}  {'cpu-ref':>9s}  {'speedup':>8s}")
for label, T, mq, f_ms, c_ms, sp in results:
    print(f"{label:32s}  {T:6d}  {mq:6d}  {f_ms:6.3f}ms  {c_ms:6.3f}ms  {sp:6.2f}x")
