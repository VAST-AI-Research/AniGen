"""
mtldiffrast.torch — PyTorch interface for Metal differentiable rasterization.

Usage:
    import mtldiffrast.torch as dr
    ctx = dr.MtlRasterizeContext()
    rast, rast_db = dr.rasterize(ctx, pos, tri, resolution=[H, W])
    out, _ = dr.interpolate(attr, rast, tri)
    tex_out = dr.texture(tex, uv)
    aa_out = dr.antialias(color, rast, pos, tri)
"""
from .ops import (
    MtlRasterizeContext,
    DepthPeeler,
    rasterize,
    interpolate,
    texture,
    texture_construct_mip,
    antialias,
    antialias_construct_topology_hash,
    get_log_level,
    set_log_level,
)
