# mtldiffrast

Metal implementation of differentiable rasterization primitives for Apple Silicon.

Based on [Modular Primitives for High-Performance Differentiable Rendering](https://arxiv.org/abs/2011.03277) (Laine et al. 2020).

## Features

- **Rasterize**: Triangle rasterization with differentiable barycentrics
- **Interpolate**: Vertex attribute interpolation with screen-space derivatives
- **Texture**: Texture sampling with mipmapping (nearest, linear, trilinear)
- **Antialias**: Differentiable antialiasing for silhouette edges
- **Zero-copy**: MPS and CPU tensors bind directly to Metal via unified memory

All operations support `torch.autograd` for end-to-end differentiable rendering.

API-compatible with [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for drop-in use in existing pipelines.

## Installation

Requires macOS with Apple Silicon, Xcode command line tools, and PyTorch >= 2.0.

```bash
pip install --no-build-isolation -e .
python setup.py build_ext --inplace
```

## Usage

```python
import mtldiffrast.torch as dr

ctx = dr.MtlRasterizeContext()
rast, rast_db = dr.rasterize(ctx, pos, tri, resolution=[512, 512])
out, _ = dr.interpolate(attr, rast, tri)
tex_out = dr.texture(tex, uv)
aa_out = dr.antialias(color, rast, pos, tri)
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
