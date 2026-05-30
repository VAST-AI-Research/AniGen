"""Benchmark: mtldiffrast MPS zero-copy input vs old CPU-bounce.

Measures rasterize at increasing resolutions with MPS input tensors. After
the 2C fix, MPS inputs wrap zero-copy via at::native::mps::getMTLBufferStorage;
before, tensor_to_mtl_buffer did t.cpu() on every call.
"""
import time
import torch
import mtldiffrast

assert torch.backends.mps.is_available()


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
    print(f"  {label:56s} {ms:8.3f} ms/call")
    return ms


print("=" * 78)
print("mtldiffrast MPS benchmark (M3 Max)")
print("=" * 78)

ctx = mtldiffrast.MtlRasterizeContext()

# Build a moderately-sized triangle mesh
def mesh(n_tris=500):
    torch.manual_seed(0)
    n_verts = n_tris * 3 // 2
    pos = torch.randn(1, n_verts, 4) * 0.5
    pos[..., 3] = 1.0  # w=1
    tri = torch.randint(0, n_verts, (n_tris, 3), dtype=torch.int32)
    return pos, tri


for n_tris in [200, 2000, 10000]:
    for res in [128, 256, 512]:
        pos_cpu, tri_cpu = mesh(n_tris)
        pos_m = pos_cpu.to("mps")
        tri_m = tri_cpu.to("mps")

        def mps_call():
            r, db = mtldiffrast.rasterize(ctx, pos_m, tri_m, resolution=[res, res])
            return r

        def cpu_bounce():
            # Simulates pre-2C behavior: force input through CPU every call.
            r, db = mtldiffrast.rasterize(ctx, pos_m.cpu(), tri_m.cpu(), resolution=[res, res])
            return r

        def cpu_native():
            r, db = mtldiffrast.rasterize(ctx, pos_cpu, tri_cpu, resolution=[res, res])
            return r

        print(f"\nn_tris={n_tris}  resolution={res}x{res}")
        mps_ms = bench("rasterize MPS (zero-copy input)", mps_call)
        bounce_ms = bench("rasterize CPU-bounce (pre-2C)", cpu_bounce, mps_sync=False)
        cpu_ms = bench("rasterize CPU native", cpu_native, mps_sync=False)
        print(f"  speedup MPS vs CPU-bounce: {bounce_ms / mps_ms:.2f}x")
