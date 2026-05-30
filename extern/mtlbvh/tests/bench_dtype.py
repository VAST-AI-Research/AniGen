"""Benchmark: mtlbvh scalar-output dtype specializations (fp32/fp16/bf16).

Measures per-dtype query time for the four kernels at realistic shapes
on both CPU and MPS device paths. Internal BVH traversal math runs fp32
on every path — only the final scalar store narrows — so any speedup
comes from half-bandwidth on the output buffer and (more importantly)
from downstream consumers being able to read fp16/bf16 directly without
an explicit upcast.

Usage: python tests/bench_dtype.py
"""
import sys
import time
import torch
import mtlbvh


def make_mesh(n_tris, seed=42):
    torch.manual_seed(seed)
    n_verts = n_tris * 3 // 2
    verts = (torch.randn(n_verts, 3) * 5).float()
    tris = torch.randint(0, n_verts, (n_tris, 3), dtype=torch.int32)
    return verts, tris


def bench(fn, warmup, iters, mps_sync=False):
    for _ in range(warmup):
        fn()
    if mps_sync:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if mps_sync:
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_NAMES  = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def _run_one(label, callers, iters, mps_sync):
    times = {dt: bench(c, warmup=1, iters=iters, mps_sync=mps_sync)
             for dt, c in callers.items()}
    row = f"  {label:30s}" + "".join(f"{times[d]:>9.2f}ms" for d in _DTYPES)
    row += f"   fp16/fp32={times[torch.float16]/times[torch.float32]:.2f}×"
    row += f"  bf16/fp32={times[torch.bfloat16]/times[torch.float32]:.2f}×"
    print(row, flush=True)


def main():
    print("=" * 78, flush=True)
    print("mtlbvh dtype specialization bench — M3 Max", flush=True)
    print("=" * 78, flush=True)

    verts, tris = make_mesh(n_tris=20_000)
    bvh = mtlbvh.MtlBVH(verts, tris)
    print(f"BVH over {tris.shape[0]} triangles\n", flush=True)
    print(f"{'  kernel':32s}" + "".join(f"{_NAMES[d]:>11s}" for d in _DTYPES) + "       ratio vs fp32", flush=True)

    has_mps = torch.backends.mps.is_available()
    devices = ["cpu"] + (["mps"] if has_mps else [])

    for dev in devices:
        for N_query in [10_000, 100_000]:
            torch.manual_seed(0)
            pts = (torch.randn(N_query, 3) * 3).float().to(dev)
            ro  = (torch.randn(N_query, 3) * 3).float().to(dev)
            rd  = torch.randn(N_query, 3).float()
            rd  = (rd / rd.norm(dim=-1, keepdim=True)).to(dev)

            print(f"\n--- device={dev}  N_query={N_query:,}", flush=True)

            mps_sync = (dev == "mps")
            _run_one("unsigned_distance", {
                dt: (lambda dt=dt: bvh.unsigned_distance(pts, output_dtype=dt))
                for dt in _DTYPES
            }, iters=20, mps_sync=mps_sync)

            _run_one("signed_distance (watertight)", {
                dt: (lambda dt=dt: bvh.signed_distance(pts, mode="watertight", output_dtype=dt))
                for dt in _DTYPES
            }, iters=10, mps_sync=mps_sync)

            # raystab is ~32× more expensive; cap at smaller N for tractable runtime.
            n_rs = min(N_query, 10_000)
            pts_rs = pts[:n_rs]
            _run_one(f"signed_distance (raystab, N={n_rs})", {
                dt: (lambda dt=dt: bvh.signed_distance(pts_rs, mode="raystab", output_dtype=dt))
                for dt in _DTYPES
            }, iters=5, mps_sync=mps_sync)

            _run_one("ray_trace", {
                dt: (lambda dt=dt: bvh.ray_trace(ro, rd, output_dtype=dt))
                for dt in _DTYPES
            }, iters=20, mps_sync=mps_sync)


if __name__ == "__main__":
    main()
