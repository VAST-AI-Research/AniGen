"""Wall-clock benchmark for the Metal spconv on MPS at trellis2-relevant shapes.

Reports per-call latency for the new MPS-native flex_gemm path, and contrasts
with the old "force-to-CPU and wait" behavior the previous mtlgemm shipped
(emulated by setting FLEX_GEMM_AUTOTUNE=1, which re-enables the per-call sync
on the CPU dispatch path; we additionally force CPU tensors into the call to
mimic what the old code was doing on every input).
"""

import os
import sys
import time
import torch

assert torch.backends.mps.is_available(), "Need MPS"

DEVICE = "mps"


def sphere_coords(res: int, ch: int, dtype: torch.dtype, device: str):
    coords = torch.stack(torch.meshgrid(
        torch.arange(res), torch.arange(res), torch.arange(res), indexing="ij",
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.contiguous().to(device)
    feats = torch.randn(coords.shape[0], ch, dtype=dtype).to(device).contiguous()
    return feats, coords, torch.Size([1, ch, res, res, res])


def bench(label, fn, num_warmup=5, num_iter=50, sync_fn=None):
    for _ in range(num_warmup):
        fn()
    if sync_fn:
        sync_fn()
    t0 = time.perf_counter()
    for _ in range(num_iter):
        fn()
    if sync_fn:
        sync_fn()
    elapsed = time.perf_counter() - t0
    per_call_ms = (elapsed / num_iter) * 1000.0
    print(f"  {label:48s} {per_call_ms:8.3f} ms/call")
    return per_call_ms


def run_shape(res, ch, dtype):
    import flex_gemm
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm

    Co = ch
    Ks = 3
    V = Ks ** 3

    feats_m, coords_m, shape = sphere_coords(res, ch, dtype, "mps")
    weight_m = torch.randn(Co, Ks, Ks, Ks, ch, dtype=dtype).to("mps")
    bias_m = torch.randn(Co, dtype=dtype).to("mps")
    N = feats_m.shape[0]

    # Build dense cache for the dense path.
    set_algorithm(Algorithm.IMPLICIT_GEMM)
    dense_out, cache_dense = sparse_submanifold_conv3d(feats_m, coords_m, shape, weight_m, bias_m)

    # Build masked cache (different cache schema — gray_code, sorted_idx, etc.)
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
    masked_out, cache_masked = sparse_submanifold_conv3d(feats_m, coords_m, shape, weight_m, bias_m)

    # Numerical parity check inside the bench so a regression here is loud.
    diff = (masked_out.detach().cpu().float() - dense_out.detach().cpu().float()).abs().max().item()
    parity_tol = {torch.float16: 2e-2, torch.bfloat16: 5e-2, torch.float32: 1e-4}[dtype]
    parity_ok = diff <= parity_tol
    parity_msg = "OK" if parity_ok else f"FAIL diff={diff:.4e} tol={parity_tol:.4e}"

    # CPU path (mimics the OLD broken code's behavior — every input went to CPU.)
    feats_c = feats_m.cpu().contiguous()
    coords_c = coords_m.cpu().contiguous()
    weight_c = weight_m.cpu().contiguous()
    bias_c = bias_m.cpu().contiguous()
    set_algorithm(Algorithm.IMPLICIT_GEMM)
    _, cache_c = sparse_submanifold_conv3d(feats_c, coords_c, shape, weight_c, bias_c)

    print(f"\nshape: res={res}, ch={ch}, dtype={dtype}, N={N}, V={V}  parity[masked vs dense]={parity_msg}")

    def _mps_dense():
        set_algorithm(Algorithm.IMPLICIT_GEMM)
        sparse_submanifold_conv3d(feats_m, coords_m, shape, weight_m, bias_m, neighbor_cache=cache_dense)

    def _mps_masked():
        set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
        sparse_submanifold_conv3d(feats_m, coords_m, shape, weight_m, bias_m, neighbor_cache=cache_masked)

    def _cpu_old_path():
        # Simulate the old broken behavior: take MPS inputs, drop to CPU, run
        # kernel on CPU memory, return CPU tensor — what every spconv call did.
        set_algorithm(Algorithm.IMPLICIT_GEMM)
        feats_round = feats_m.cpu().contiguous()
        weight_round = weight_m.cpu().contiguous()
        bias_round = bias_m.cpu().contiguous()
        out, _ = sparse_submanifold_conv3d(feats_round, coords_c, shape,
                                            weight_round, bias_round, neighbor_cache=cache_c)
        out.to("mps")

    def _cpu_native():
        set_algorithm(Algorithm.IMPLICIT_GEMM)
        sparse_submanifold_conv3d(feats_c, coords_c, shape, weight_c, bias_c, neighbor_cache=cache_c)

    dense_ms = bench("flex_gemm MPS dense implicit GEMM",     _mps_dense,    sync_fn=torch.mps.synchronize)
    mskd_ms  = bench("flex_gemm MPS masked implicit GEMM",    _mps_masked,   sync_fn=torch.mps.synchronize)
    old_ms   = bench("flex_gemm OLD CPU-bounce path",          _cpu_old_path, sync_fn=torch.mps.synchronize)
    cpu_ms   = bench("flex_gemm CPU-only (best CPU case)",     _cpu_native)

    speedup_vs_old   = old_ms / mskd_ms
    speedup_vs_dense = dense_ms / mskd_ms
    print(f"  speedup masked vs dense (MPS):     {speedup_vs_dense:.2f}x")
    print(f"  speedup masked vs old CPU-bounce:  {speedup_vs_old:.2f}x")
    return dense_ms, mskd_ms, old_ms, cpu_ms, parity_ok


print("=" * 75)
print("mtlgemm Metal spconv benchmark on MPS")
print(f"PyTorch {torch.__version__}, MPS device: {torch.mps.current_allocated_memory() // 1024**2}MB")
print("=" * 75)
results = []
# Trellis2-decoder-relevant shapes (last two stress the wide-tile variant added in Phase C)
for res, ch, dtype in [
    (16, 64, torch.float16),
    (32, 64, torch.float16),
    (32, 128, torch.float16),
    (64, 128, torch.float16),
    (64, 256, torch.float16),
    (128, 256, torch.float16),
]:
    dense, mskd, old, cpu, parity = run_shape(res, ch, dtype)
    results.append((res, ch, dtype, dense, mskd, old, cpu, parity))

print("\n" + "=" * 75)
print("Summary (lower is better)")
print("=" * 75)
print(f"{'shape':28s}  {'dense':>9s}  {'masked':>9s}  {'mps-old':>9s}  {'cpu':>9s}  {'msk/dense':>10s}  {'msk/old':>8s}  {'parity':>7s}")
for res, ch, dt, d, m, o, c, parity in results:
    print(f"res={res:3d} ch={ch:3d} dt={str(dt).split('.')[-1]:8s}  "
          f"{d:6.3f}ms  {m:6.3f}ms  {o:6.3f}ms  {c:6.3f}ms  "
          f"{m/d:8.2f}x  {o/m:6.2f}x  {('OK' if parity else 'FAIL'):>7s}")
