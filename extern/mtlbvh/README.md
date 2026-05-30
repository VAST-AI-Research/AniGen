# mtlbvh

Metal-accelerated BVH for distance queries and ray tracing on Apple Silicon.

## Features

- **Unsigned distance**: closest point on mesh surface
- **Signed distance**: watertight (normal averaging) and raystab (ray stabbing) modes
- **Ray tracing**: ray-triangle intersection with BVH acceleration
- **Barycentric coordinates**: optional UVW output for all distance queries
- **Zero-copy**: tensors bind directly to Metal via Apple Silicon unified memory

4-wide BVH with stack-based traversal, sorting networks, and stackless fallback.

API-compatible with [cubvh](https://github.com/ashawkey/cubvh) for drop-in use in existing pipelines.

## Installation

Requires macOS with Apple Silicon, Xcode command line tools, and PyTorch >= 2.0.

```bash
pip install --no-build-isolation -e .
python setup.py build_ext --inplace
```

## Usage

```python
from mtlbvh import MtlBVH

bvh = MtlBVH(vertices, triangles)  # numpy or torch, [N,3] and [M,3]

# distance queries
distances, face_id, uvw = bvh.unsigned_distance(positions, return_uvw=True)
distances, face_id, uvw = bvh.signed_distance(positions, mode='watertight')

# ray tracing
hit_pos, face_id, depth = bvh.ray_trace(rays_o, rays_d)
```

## License

MIT License

---

## Apple-Silicon follow-ups

Cross-repo follow-up work (deferred mtlgemm perf items, fp16/bf16 shader
specializations, universal tiled sparse attention, etc.) is tracked in
`/Users/gusto/work/ai/FOLLOWUPS.md` alongside the four `mtl*` repos. If
you touch any of them, update that file — it's the single source of truth
for what's intentionally deferred vs what's still broken.
