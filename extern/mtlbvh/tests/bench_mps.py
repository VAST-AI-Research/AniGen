"""Benchmark: mtlbvh MPS path vs CPU path + vs old CPU-bounce behavior.

Measures the three query kernels (unsigned_distance, signed_distance,
ray_trace) on M3 Max across input/output device choices.
"""
import time
import torch
import mtlbvh

assert torch.backends.mps.is_available()


def make_mesh(n_tris=5000, seed=42):
    """Random triangulated blob (not a real mesh, just a stress shape)."""
    torch.manual_seed(seed)
    n_verts = n_tris * 3 // 2  # rough triangle-to-vertex ratio
    verts = (torch.randn(n_verts, 3) * 5).float()
    tris = torch.randint(0, n_verts, (n_tris, 3), dtype=torch.int32)
    return verts, tris


def bench(label, fn, warmup=3, iters=20, mps_sync=False):
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
print("mtlbvh MPS benchmark (M3 Max)")
print("=" * 78)

verts, tris = make_mesh(n_tris=5000)
bvh = mtlbvh.MtlBVH(verts, tris)

for N_query in [1000, 10000, 100000]:
    pts = (torch.randn(N_query, 3) * 3).float()

    # MPS path (after 2B fix)
    pts_m = pts.to("mps")

    def mps_call():
        d, f, _ = bvh.unsigned_distance(pts_m)
        return d

    # Old CPU-bounce: force input to CPU (simulating pre-2B behavior where
    # tensor_to_buffer did .cpu() internally), keep result whatever the kernel
    # returns (which would have been CPU too).
    def cpu_bounce():
        d, f, _ = bvh.unsigned_distance(pts_m.cpu())
        return d.to("mps") if d.device.type == "cpu" else d

    def cpu_native():
        d, f, _ = bvh.unsigned_distance(pts)
        return d

    print(f"\nN_query={N_query}")
    mps_ms = bench("unsigned_distance MPS (zero-copy)", mps_call, mps_sync=True)
    bounce_ms = bench("unsigned_distance CPU-bounce (pre-2B behavior)", cpu_bounce)
    cpu_ms = bench("unsigned_distance CPU native", cpu_native)
    print(f"  speedup MPS vs CPU-bounce: {bounce_ms / mps_ms:.2f}x")

    # Ray trace
    ro_m = pts.to("mps")
    rd_cpu = torch.randn(N_query, 3).float()
    rd_cpu = rd_cpu / rd_cpu.norm(dim=-1, keepdim=True)
    rd_m = rd_cpu.to("mps")

    def rt_mps():
        return bvh.ray_trace(ro_m, rd_m)

    def rt_bounce():
        return bvh.ray_trace(ro_m.cpu(), rd_m.cpu())

    rt_mps_ms = bench("ray_trace MPS", rt_mps, mps_sync=True)
    rt_bounce_ms = bench("ray_trace CPU-bounce", rt_bounce)
    print(f"  speedup ray_trace MPS vs CPU-bounce: {rt_bounce_ms / rt_mps_ms:.2f}x")
