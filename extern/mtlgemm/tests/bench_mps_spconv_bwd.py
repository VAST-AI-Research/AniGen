"""Backward benchmark for masked vs dense implicit GEMM on MPS.

Measures the bwd staticmethod directly (same code path trellis2 hits
under autograd). Before this session the masked bwd aliased to the dense
bwd; after, it uses real masked kernels for grad_input (valid_kernel)
and grad_weight (valid_signal_*).
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")
import time
import torch

assert torch.backends.mps.is_available()

from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction


def sphere_coords(res, ch, dtype):
    coords = torch.stack(torch.meshgrid(
        torch.arange(res), torch.arange(res), torch.arange(res), indexing="ij",
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.to("mps").contiguous()
    feats = torch.randn(coords.shape[0], ch, dtype=dtype).to("mps").contiguous()
    return feats, coords, torch.Size([1, ch, res, res, res])


def run_bwd(algo, feats, coords, shape, weight, bias, grad_out):
    set_algorithm(algo)
    f = feats.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    Co, Kw, Kh, Kd, Ci = w.shape
    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), (1,1,1))
    def call():
        set_algorithm(algo)  # re-assert — global setting may flip between lambdas
        return SubMConv3dFunction._sparse_submanifold_conv_backward(
            grad_out.contiguous(), f, cache, w, b,
        )
    return call


def bench_bwd(label, fn, warm=3, iters=20):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:48s} {ms:7.3f} ms/call")
    return ms


print("=" * 70)
print("Backward-masked GEMM benchmark on MPS (M3 Max, fp16)")
print("=" * 70)
for res, ch in [(16, 64), (32, 64), (32, 128), (64, 128), (64, 256)]:
    dtype = torch.float16
    feats, coords, shape = sphere_coords(res, ch, dtype)
    weight = torch.randn(ch, 3, 3, 3, ch, dtype=dtype).to("mps")
    bias = torch.randn(ch, dtype=dtype).to("mps")
    grad_out = torch.randn(feats.shape[0], ch, dtype=dtype).to("mps")
    N = feats.shape[0]
    print(f"\nshape: res={res} ch={ch} N={N}")
    dense = run_bwd(Algorithm.IMPLICIT_GEMM, feats, coords, shape, weight, bias, grad_out)
    masked = run_bwd(Algorithm.MASKED_IMPLICIT_GEMM, feats, coords, shape, weight, bias, grad_out)
    d_ms = bench_bwd("dense bwd (grad_input + grad_weight)", dense)
    m_ms = bench_bwd("masked bwd (grad_input + grad_weight)", masked)
    print(f"  speedup masked/dense: {d_ms/m_ms:.2f}x")
