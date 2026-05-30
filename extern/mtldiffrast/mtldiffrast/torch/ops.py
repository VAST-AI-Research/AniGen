"""
mtldiffrast.torch.ops — Differentiable rasterization operations for PyTorch.

Operations: rasterize, interpolate, texture, antialias.
All operations support differentiable rendering via torch.autograd.Function.
Zero-copy MPS path: tensors stay on device, no CPU roundtrip needed.
"""
import numpy as np
import torch
from .. import _C


class MtlRasterizeContext:
    """Metal rasterization context."""
    def __init__(self, device=None):
        self._ctx = _C.MtlRasterizeContext()
        self.active_depth_peeler = None



# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------

class _rasterize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, glctx, pos, tri, resolution, grad_db):
        pos_f = pos.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()
        rast_out, rast_db = _C.rasterize_fwd(glctx._ctx, pos_f, tri_i, list(resolution))
        ctx.save_for_backward(pos, tri, rast_out, rast_db)
        ctx.saved_glctx = glctx
        ctx.saved_grad_db = grad_db
        return rast_out.unsqueeze(0), rast_db.unsqueeze(0)

    @staticmethod
    def backward(ctx, dy, ddb):
        pos, tri, rast_out, rast_db = ctx.saved_tensors
        glctx = ctx.saved_glctx

        dy_2d = dy[0].contiguous().float()
        ddb_2d = ddb[0].contiguous().float() if ctx.saved_grad_db else torch.empty(0)

        pos_f = pos.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()

        g_pos = _C.rasterize_grad(glctx._ctx, pos_f, tri_i,
                                   rast_out.contiguous(), dy_2d, ddb_2d)

        if pos.dim() == 3:
            g_pos = g_pos.unsqueeze(0)
        g_pos = g_pos.to(pos.device)

        return None, g_pos, None, None, None


class _interpolate_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attr, rast, tri, rast_db):
        attr_2d = attr[0] if attr.dim() == 3 else attr
        rast_2d = rast[0] if rast.dim() == 4 else rast

        attr_f = attr_2d.detach().contiguous().float()
        rast_f = rast_2d.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()
        rast_db_empty = torch.empty(0)

        out, out_da = _C.interpolate_fwd(attr_f, rast_f, tri_i, rast_db_empty, False)

        ctx.save_for_backward(attr_2d, rast_2d, tri)
        ctx.attr_batched = (attr.dim() == 3)
        ctx.rast_batched = (rast.dim() == 4)
        out = out.unsqueeze(0)
        H, W = rast_2d.shape[0], rast_2d.shape[1]
        out_da = torch.empty([1, H, W, 0], dtype=torch.float32)
        return out, out_da

    @staticmethod
    def backward(ctx, dy, _dda):
        attr, rast, tri = ctx.saved_tensors

        dy_2d = dy[0].contiguous().float()
        attr_f = attr.detach().contiguous().float()
        rast_f = rast.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()

        g_attr, g_rast = _C.interpolate_grad(attr_f, rast_f, tri_i, dy_2d)

        g_attr_out = g_attr.unsqueeze(0) if ctx.attr_batched else g_attr
        g_rast_out = g_rast.unsqueeze(0) if ctx.rast_batched else g_rast

        return g_attr_out.to(dy.device), g_rast_out.to(dy.device), None, None


class _interpolate_func_da(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attr, rast, tri, rast_db, diff_attrs):
        attr_2d = attr[0] if attr.dim() == 3 else attr
        rast_2d = rast[0] if rast.dim() == 4 else rast

        attr_f = attr_2d.detach().contiguous().float()
        rast_f = rast_2d.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()
        rast_db_f = rast_db[0].detach().contiguous().float() if rast_db.dim() == 4 else rast_db.detach().contiguous().float()

        out, out_da = _C.interpolate_fwd(attr_f, rast_f, tri_i, rast_db_f, True)

        ctx.save_for_backward(attr_2d, rast_2d, tri, rast_db[0] if rast_db.dim() == 4 else rast_db)
        out = out.unsqueeze(0)
        if out_da is not None and out_da.numel() > 0:
            out_da = out_da.unsqueeze(0)
        else:
            H, W = rast_2d.shape[0], rast_2d.shape[1]
            out_da = torch.empty([1, H, W, 0], dtype=torch.float32)
        return out, out_da

    @staticmethod
    def backward(ctx, dy, dda):
        attr, rast, tri, rast_db = ctx.saved_tensors

        dy_2d = dy[0].contiguous().float()
        attr_f = attr.detach().contiguous().float()
        rast_f = rast.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()

        if dda is not None and dda.numel() > 0:
            dda_2d = dda[0].contiguous().float()
            H, W = rast_f.shape[0], rast_f.shape[1]
            A = attr_f.shape[1]
            dda_reshaped = dda_2d.view(H, W, A, 2)
            rast_db_f = rast_db.detach().contiguous().float()
            g_attr, g_rast, g_rast_db = _C.interpolate_grad_da(
                attr_f, rast_f, tri_i, dy_2d, rast_db_f, dda_reshaped)
            g_rast_db = g_rast_db.unsqueeze(0) if rast_db.dim() == 3 else g_rast_db
        else:
            g_attr, g_rast = _C.interpolate_grad(attr_f, rast_f, tri_i, dy_2d)
            g_rast_db = None

        g_attr_out = g_attr.unsqueeze(0) if attr.dim() == 3 else g_attr
        g_rast_out = g_rast.unsqueeze(0)

        return g_attr_out.to(attr.device), g_rast_out.to(rast.device), None, g_rast_db, None


class _texture_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum):
        tex_f = tex.detach().contiguous().float()
        uv_f = uv.detach().contiguous().float()
        out = _C.texture_fwd(tex_f, uv_f, filter_mode_enum, boundary_mode_enum)
        ctx.save_for_backward(tex, uv)
        ctx.saved_misc = (filter_mode, filter_mode_enum, boundary_mode_enum)
        return out

    @staticmethod
    def backward(ctx, dy):
        tex, uv = ctx.saved_tensors
        filter_mode, filter_mode_enum, boundary_mode_enum = ctx.saved_misc

        tex_f = tex.detach().contiguous().float()
        uv_f = uv.detach().contiguous().float()
        dy_f = dy.detach().contiguous().float()

        mip_wrapper = _C.TextureMipWrapper()
        g_tex, g_uv, _, _ = _C.texture_grad(
            tex_f, uv_f, dy_f,
            torch.empty(0), torch.empty(0),
            mip_wrapper, [],
            filter_mode_enum, boundary_mode_enum)

        if filter_mode == 'nearest':
            return None, g_tex.to(tex.device), None, None, None
        return None, g_tex.to(tex.device), g_uv.to(uv.device), None, None


class _texture_func_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, filter_mode, tex, uv, uv_da, mip_level_bias,
                mip_wrapper, mip_stack, filter_mode_enum, boundary_mode_enum):
        tex_f = tex.detach().contiguous().float()
        uv_f = uv.detach().contiguous().float()
        uv_da_f = uv_da.detach().contiguous().float() if uv_da is not None else torch.empty(0)
        mip_bias_f = mip_level_bias.detach().contiguous().float() if mip_level_bias is not None else torch.empty(0)
        mip_stack_f = [m.detach().contiguous().float() for m in mip_stack] if mip_stack else []

        out = _C.texture_fwd_mip(tex_f, uv_f, uv_da_f, mip_bias_f,
                                  mip_wrapper, mip_stack_f,
                                  filter_mode_enum, boundary_mode_enum)

        save_list = [tex, uv]
        if uv_da is not None:
            save_list.append(uv_da)
        if mip_level_bias is not None:
            save_list.append(mip_level_bias)
        ctx.save_for_backward(*save_list)
        ctx.saved_misc = (filter_mode, mip_wrapper, mip_stack_f,
                          filter_mode_enum, boundary_mode_enum,
                          uv_da is not None, mip_level_bias is not None)
        return out

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        filter_mode, mip_wrapper, mip_stack_f, filter_mode_enum, boundary_mode_enum, has_uv_da, has_mip_bias = ctx.saved_misc

        tex = saved[0]
        uv = saved[1]
        idx = 2
        uv_da = saved[idx] if has_uv_da else None
        if has_uv_da: idx += 1
        mip_level_bias = saved[idx] if has_mip_bias else None

        tex_f = tex.detach().contiguous().float()
        uv_f = uv.detach().contiguous().float()
        dy_f = dy.detach().contiguous().float()
        uv_da_f = uv_da.detach().contiguous().float() if uv_da is not None else torch.empty(0)
        mip_bias_f = mip_level_bias.detach().contiguous().float() if mip_level_bias is not None else torch.empty(0)

        g_tex, g_uv, g_uv_da, g_mip_bias = _C.texture_grad(
            tex_f, uv_f, dy_f,
            uv_da_f, mip_bias_f,
            mip_wrapper, mip_stack_f,
            filter_mode_enum, boundary_mode_enum)

        g_uv_da_out = g_uv_da.to(uv_da.device) if g_uv_da is not None and uv_da is not None else None
        g_mip_out = g_mip_bias.to(mip_level_bias.device) if g_mip_bias is not None and mip_level_bias is not None else None

        return (None, g_tex.to(tex.device), g_uv.to(uv.device),
                g_uv_da_out, g_mip_out, None, None, None, None)


class _antialias_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, color, rast, pos, tri, topology_hash, pos_gradient_boost):
        color_f = color.detach().contiguous().float()
        rast_f = rast.detach().contiguous().float()
        pos_f = pos.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()

        if topology_hash is None:
            topology_hash = _C.TopologyHashWrapper()
            _C.antialias_construct_topology_hash(tri_i, topology_hash)

        out, work_buffer = _C.antialias_fwd(color_f, rast_f, pos_f, tri_i, topology_hash)

        ctx.save_for_backward(color, rast, pos, tri)
        ctx.saved_misc = (pos_gradient_boost, work_buffer)
        return out

    @staticmethod
    def backward(ctx, dy):
        color, rast, pos, tri = ctx.saved_tensors
        pos_gradient_boost, work_buffer = ctx.saved_misc

        color_f = color.detach().contiguous().float()
        rast_f = rast.detach().contiguous().float()
        pos_f = pos.detach().contiguous().float()
        tri_i = tri.detach().contiguous().int()
        dy_f = dy.detach().contiguous().float()

        g_color, g_pos = _C.antialias_grad(color_f, rast_f, pos_f, tri_i, dy_f, work_buffer)

        if pos_gradient_boost != 1.0:
            g_pos = g_pos * pos_gradient_boost

        return g_color.to(color.device), None, g_pos.to(pos.device), None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    """Rasterize triangles using Metal. Supports backward pass."""
    assert isinstance(glctx, MtlRasterizeContext)
    assert isinstance(pos, torch.Tensor) and isinstance(tri, torch.Tensor)
    resolution = tuple(resolution)

    if glctx.active_depth_peeler is not None:
        raise RuntimeError("Cannot call rasterize() during depth peeling, use rasterize_next_layer()")

    # Handle batched positions
    if pos.dim() == 3:
        pos = pos[0]
    assert pos.dim() == 2 and pos.size(1) == 4

    return _rasterize_func.apply(glctx, pos, tri, resolution, grad_db)


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Interpolate vertex attributes. Supports backward pass."""
    assert all(isinstance(x, torch.Tensor) for x in (attr, rast, tri))

    enable_da = diff_attrs is not None and rast_db is not None

    if enable_da:
        return _interpolate_func_da.apply(attr, rast, tri, rast_db, diff_attrs)
    else:
        return _interpolate_func.apply(attr, rast, tri, rast_db)


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Perform texture sampling. Supports backward pass."""
    if filter_mode == 'auto':
        filter_mode = 'linear-mipmap-linear' if (uv_da is not None or mip_level_bias is not None) else 'linear'

    if max_mip_level is None:
        max_mip_level = -1
    else:
        max_mip_level = int(max_mip_level)

    if max_mip_level == 0 and filter_mode in ['linear-mipmap-nearest', 'linear-mipmap-linear']:
        filter_mode = 'linear'

    filter_mode_dict = {'nearest': 0, 'linear': 1, 'linear-mipmap-nearest': 2, 'linear-mipmap-linear': 3}
    boundary_mode_dict = {'cube': 0, 'wrap': 1, 'clamp': 2, 'zero': 3}
    filter_mode_enum = filter_mode_dict[filter_mode]
    boundary_mode_enum = boundary_mode_dict[boundary_mode]

    assert isinstance(tex, torch.Tensor) and isinstance(uv, torch.Tensor)

    use_mip = filter_mode_enum >= 2

    if use_mip:
        mip_stack = []
        if mip is not None:
            if isinstance(mip, list):
                mip_stack = [m.detach().contiguous().float() for m in mip]

        mip_wrapper = _C.TextureMipWrapper()
        mip_wrapper.max_mip_level = max_mip_level
        mip_wrapper.cube_mode = (boundary_mode == 'cube')

        return _texture_func_mip.apply(filter_mode, tex, uv, uv_da, mip_level_bias,
                                        mip_wrapper, mip_stack,
                                        filter_mode_enum, boundary_mode_enum)
    else:
        return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)


def texture_construct_mip(tex, max_mip_level=None, cube_mode=False):
    """Construct a mipmap stack for a texture."""
    if max_mip_level is None:
        max_mip_level = -1
    tex_f = tex.detach().contiguous().float()
    return _C.texture_construct_mip(tex_f, max_mip_level, cube_mode)


# ---------------------------------------------------------------------------
# Antialias
# ---------------------------------------------------------------------------

def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Perform antialiasing. Supports backward pass."""
    assert all(isinstance(x, torch.Tensor) for x in (color, rast, pos, tri))

    if topology_hash is None:
        tri_i = tri.detach().contiguous().int()
        topology_hash = _C.TopologyHashWrapper()
        _C.antialias_construct_topology_hash(tri_i, topology_hash)

    return _antialias_func.apply(color, rast, pos, tri, topology_hash, pos_gradient_boost)


def antialias_construct_topology_hash(tri):
    """Construct a topology hash for antialias."""
    tri_i = tri.detach().contiguous().int()
    wrapper = _C.TopologyHashWrapper()
    _C.antialias_construct_topology_hash(tri_i, wrapper)
    return wrapper


# ---------------------------------------------------------------------------
# Depth peeler (compatibility stub)
# ---------------------------------------------------------------------------

class DepthPeeler:
    """Depth peeler for multi-layer rasterization."""
    def __init__(self, glctx, pos, tri, resolution, ranges=None, grad_db=True):
        assert isinstance(glctx, MtlRasterizeContext)
        self.raster_ctx = glctx
        self.pos = pos
        self.tri = tri
        self.resolution = tuple(resolution)
        self.ranges = ranges
        self.grad_db = grad_db
        self.peeling_idx = None

    def __enter__(self):
        if self.raster_ctx.active_depth_peeler is not None:
            raise RuntimeError("Cannot have multiple depth peelers active simultaneously")
        self.raster_ctx.active_depth_peeler = self
        self.peeling_idx = 0
        return self

    def __exit__(self, *args):
        self.raster_ctx.active_depth_peeler = None
        self.raster_ctx = None
        self.peeling_idx = None

    def rasterize_next_layer(self):
        assert self.raster_ctx.active_depth_peeler is self
        self.raster_ctx.active_depth_peeler = None
        try:
            result = rasterize(self.raster_ctx, self.pos, self.tri, self.resolution)
        finally:
            self.raster_ctx.active_depth_peeler = self
        self.peeling_idx += 1
        return result


# ---------------------------------------------------------------------------
# Log level (compatibility stubs)
# ---------------------------------------------------------------------------

_log_level = 1

def get_log_level():
    return _log_level

def set_log_level(level):
    global _log_level
    _log_level = level


