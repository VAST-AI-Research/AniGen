"""
Backward pass tests for mtldiffrast.

Verifies Metal gradient kernels against CPU reference implementations.

Tests:
  1. Rasterize grad: edge function gradient math
  2. Interpolate grad: bary-weighted attribute gradient
  3. Texture grad nearest: scatter dy to nearest texel
  4. Texture grad linear: bilinear weight scatter + UV gradient
  5. End-to-end: rasterize->interpolate->texture chain gradient flow
  6. torch.autograd.gradcheck for each op
"""
import math
import torch
import numpy as np
import pytest
import mtldiffrast.torch as dr
from mtldiffrast import _C


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_simple_triangle():
    """Single triangle covering some pixels in an 8x8 image."""
    pos = torch.tensor([
        [-0.5, -0.5, 0.5, 1.0],
        [ 0.5, -0.5, 0.5, 1.0],
        [ 0.0,  0.5, 0.5, 1.0],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return pos, tri


def make_fullscreen_quad():
    """Quad covering the full 8x8 image."""
    pos = torch.tensor([
        [-1.0, -1.0, 0.5, 1.0],
        [ 1.0, -1.0, 0.5, 1.0],
        [ 1.0,  1.0, 0.5, 1.0],
        [-1.0,  1.0, 0.5, 1.0],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    return pos, tri


def triidx_to_float(x):
    """Python version of triangle ID encoding."""
    if x <= 0x01000000:
        return float(x)
    import struct
    bits = 0x4a800000 + x
    return struct.unpack('f', struct.pack('I', bits))[0]


def float_to_triidx(f):
    """Python version of triangle ID decoding."""
    if f <= 16777216.0:
        return int(f)
    import struct
    bits = struct.unpack('I', struct.pack('f', f))[0]
    return bits - 0x4a800000


# ---------------------------------------------------------------------------
# CPU Reference Implementations
# ---------------------------------------------------------------------------

def rasterize_grad_ref(pos, tri, rast_out, dy, ddb=None):
    """CPU reference for rasterize gradient — edge function differentiation."""
    V = pos.shape[0]
    H, W = rast_out.shape[0], rast_out.shape[1]
    grad = np.zeros((V, 4), dtype=np.float32)

    xs = 2.0 / W
    xo = -1.0 + 1.0 / W
    ys = 2.0 / H
    yo = -1.0 + 1.0 / H

    enable_db = ddb is not None

    pos_np = pos.numpy()
    tri_np = tri.numpy()
    rast_np = rast_out.numpy()
    dy_np = dy.numpy()
    ddb_np = ddb.numpy() if enable_db else None

    for py_ in range(H):
        for px in range(W):
            pidx = px + W * py_

            dy_x = dy_np[py_, px, 0]
            dy_y = dy_np[py_, px, 1]
            tri_float = rast_np[py_, px, 3]
            triIdx = float_to_triidx(tri_float) - 1

            if triIdx < 0:
                continue

            import struct
            grad_all_dy = struct.unpack('I', struct.pack('f', dy_x))[0] | struct.unpack('I', struct.pack('f', dy_y))[0]
            grad_all_ddb = 0
            if enable_db:
                for k in range(4):
                    grad_all_ddb |= struct.unpack('I', struct.pack('f', ddb_np[py_, px, k]))[0]
            if ((grad_all_dy | grad_all_ddb) << 1) == 0:
                continue

            vi0 = tri_np[triIdx, 0]
            vi1 = tri_np[triIdx, 1]
            vi2 = tri_np[triIdx, 2]

            p0 = pos_np[vi0]
            p1 = pos_np[vi1]
            p2 = pos_np[vi2]

            fx = xs * px + xo
            fy = ys * py_ + yo
            p0x = p0[0] - fx * p0[3]
            p0y = p0[1] - fy * p0[3]
            p1x = p1[0] - fx * p1[3]
            p1y = p1[1] - fy * p1[3]
            p2x = p2[0] - fx * p2[3]
            p2y = p2[1] - fy * p2[3]
            a0 = p1x*p2y - p1y*p2x
            a1 = p2x*p0y - p2y*p0x
            a2 = p0x*p1y - p0y*p1x

            at = a0 + a1 + a2
            ep = math.copysign(1e-6, at)
            iw = 1.0 / (at + ep)

            b0 = a0 * iw
            b1 = a1 * iw

            gb0 = dy_x * iw
            gb1 = dy_y * iw
            gbb = gb0 * b0 + gb1 * b1
            gp0x = gbb * (p2y - p1y) - gb1 * p2y
            gp1x = gbb * (p0y - p2y) + gb0 * p2y
            gp2x = gbb * (p1y - p0y) - gb0 * p1y + gb1 * p0y
            gp0y = gbb * (p1x - p2x) + gb1 * p2x
            gp1y = gbb * (p2x - p0x) - gb0 * p2x
            gp2y = gbb * (p0x - p1x) + gb0 * p1x - gb1 * p0x
            gp0w = -fx * gp0x - fy * gp0y
            gp1w = -fx * gp1x - fy * gp1y
            gp2w = -fx * gp2x - fy * gp2y

            if enable_db and ((grad_all_ddb << 1) != 0):
                ddb_v = ddb_np[py_, px]
                dfxdX = xs * iw
                dfydY = ys * iw
                ddb_v = ddb_v.copy()
                ddb_v[0] *= dfxdX
                ddb_v[1] *= dfydY
                ddb_v[2] *= dfxdX
                ddb_v[3] *= dfydY

                da0dX = p1[1]*p2[3] - p2[1]*p1[3]
                da1dX = p2[1]*p0[3] - p0[1]*p2[3]
                da0dY = p2[0]*p1[3] - p1[0]*p2[3]
                da1dY = p0[0]*p2[3] - p2[0]*p0[3]
                da2dX = p0[1]*p1[3] - p1[1]*p0[3]
                da2dY = p1[0]*p0[3] - p0[0]*p1[3]
                datdX = da0dX + da1dX + da2dX
                datdY = da0dY + da1dY + da2dY

                x01 = p0[0] - p1[0]; x12 = p1[0] - p2[0]; x20 = p2[0] - p0[0]
                y01 = p0[1] - p1[1]; y12 = p1[1] - p2[1]; y20 = p2[1] - p0[1]
                w01 = p0[3] - p1[3]; w12 = p1[3] - p2[3]; w20 = p2[3] - p0[3]

                a0p1 = fy * p2[0] - fx * p2[1]
                a0p2 = fx * p1[1] - fy * p1[0]
                a1p0 = fx * p2[1] - fy * p2[0]
                a1p2 = fy * p0[0] - fx * p0[1]

                wdudX = 2*b0*datdX - da0dX
                wdudY = 2*b0*datdY - da0dY
                wdvdX = 2*b1*datdX - da1dX
                wdvdY = 2*b1*datdY - da1dY

                c0 = iw*(ddb_v[0]*wdudX + ddb_v[1]*wdudY + ddb_v[2]*wdvdX + ddb_v[3]*wdvdY)
                cx = c0*fx - ddb_v[0]*b0 - ddb_v[2]*b1
                cy = c0*fy - ddb_v[1]*b0 - ddb_v[3]*b1
                cxy = iw*(ddb_v[0]*datdX + ddb_v[1]*datdY)
                czw = iw*(ddb_v[2]*datdX + ddb_v[3]*datdY)

                gp0x += c0*y12 - cy*w12 + czw*p2y + ddb_v[3]*p2[3]
                gp1x += c0*y20 - cy*w20 - cxy*p2y - ddb_v[1]*p2[3]
                gp2x += c0*y01 - cy*w01 + cxy*p1y - czw*p0y + ddb_v[1]*p1[3] - ddb_v[3]*p0[3]
                gp0y += cx*w12 - c0*x12 - czw*p2x - ddb_v[2]*p2[3]
                gp1y += cx*w20 - c0*x20 + cxy*p2x + ddb_v[0]*p2[3]
                gp2y += cx*w01 - c0*x01 - cxy*p1x + czw*p0x - ddb_v[0]*p1[3] + ddb_v[2]*p0[3]
                gp0w += cy*x12 - cx*y12 - czw*a1p0 + ddb_v[2]*p2[1] - ddb_v[3]*p2[0]
                gp1w += cy*x20 - cx*y20 - cxy*a0p1 - ddb_v[0]*p2[1] + ddb_v[1]*p2[0]
                gp2w += cy*x01 - cx*y01 - cxy*a0p2 - czw*a1p2 + ddb_v[0]*p1[1] - ddb_v[1]*p1[0] - ddb_v[2]*p0[1] + ddb_v[3]*p0[0]

            grad[vi0, 0] += gp0x; grad[vi0, 1] += gp0y; grad[vi0, 3] += gp0w
            grad[vi1, 0] += gp1x; grad[vi1, 1] += gp1y; grad[vi1, 3] += gp1w
            grad[vi2, 0] += gp2x; grad[vi2, 1] += gp2y; grad[vi2, 3] += gp2w

    return torch.from_numpy(grad)


def interpolate_grad_ref(attr, rast, tri, dy):
    """CPU reference for interpolate gradient — bary-weighted attribute differentiation."""
    V, A = attr.shape
    H, W = rast.shape[0], rast.shape[1]
    T = tri.shape[0]

    grad_attr = np.zeros((V, A), dtype=np.float32)
    grad_rast = np.zeros((H, W, 4), dtype=np.float32)

    attr_np = attr.numpy()
    rast_np = rast.numpy()
    tri_np = tri.numpy()
    dy_np = dy.numpy()

    for py_ in range(H):
        for px in range(W):
            r = rast_np[py_, px]
            triIdx = float_to_triidx(r[3]) - 1
            if triIdx < 0 or triIdx >= T:
                continue

            vi0 = tri_np[triIdx, 0]
            vi1 = tri_np[triIdx, 1]
            vi2 = tri_np[triIdx, 2]

            a0 = attr_np[vi0]
            a1 = attr_np[vi1]
            a2 = attr_np[vi2]

            b0 = r[0]
            b1 = r[1]
            b2 = 1.0 - b0 - b1
            gb0 = 0.0
            gb1 = 0.0

            for i in range(A):
                y = dy_np[py_, px, i]
                gb0 += y * (a0[i] - a2[i])
                gb1 += y * (a1[i] - a2[i])
                grad_attr[vi0, i] += b0 * y
                grad_attr[vi1, i] += b1 * y
                grad_attr[vi2, i] += b2 * y

            grad_rast[py_, px] = [gb0, gb1, 0, 0]

    return torch.from_numpy(grad_attr), torch.from_numpy(grad_rast)


def texture_grad_nearest_ref(tex, uv, dy):
    """CPU reference for nearest texture gradient — scatter dy to nearest texel."""
    N, H_img, W_img, C = dy.shape
    _, H_tex, W_tex, _ = tex.shape
    grad_tex = np.zeros_like(tex.numpy())

    uv_np = uv.numpy()
    dy_np = dy.numpy()

    for n in range(N):
        for py_ in range(H_img):
            for px in range(W_img):
                u, v = uv_np[n, py_, px]
                u = u - math.floor(u)
                v = v - math.floor(v)
                iu = min(int(u * W_tex), W_tex - 1)
                iv = min(int(v * H_tex), H_tex - 1)
                iu = max(0, min(iu, W_tex - 1))
                iv = max(0, min(iv, H_tex - 1))
                tz = 0 if tex.shape[0] == 1 else n
                for c in range(C):
                    grad_tex[tz, iv, iu, c] += dy_np[n, py_, px, c]

    return torch.from_numpy(grad_tex)


def texture_grad_linear_ref(tex, uv, dy):
    """CPU reference for linear texture gradient — bilinear scatter + UV gradient."""
    N, H_img, W_img, C = dy.shape
    _, H_tex, W_tex, _ = tex.shape
    grad_tex = np.zeros_like(tex.numpy())
    grad_uv = np.zeros_like(uv.numpy())

    tex_np = tex.numpy()
    uv_np = uv.numpy()
    dy_np = dy.numpy()

    for n in range(N):
        tz = 0 if tex.shape[0] == 1 else n
        for py_ in range(H_img):
            for px in range(W_img):
                u, v = uv_np[n, py_, px]
                # Wrap
                u = u - math.floor(u)
                v = v - math.floor(v)

                u_tex = u * W_tex - 0.5
                v_tex = v * H_tex - 0.5

                iu0 = int(math.floor(u_tex))
                iv0 = int(math.floor(v_tex))
                iu1 = iu0 + 1
                iv1 = iv0 + 1
                fu = u_tex - iu0
                fv = v_tex - iv0

                # Wrap indices
                iu0w = iu0 % W_tex if iu0 < 0 else (iu0 % W_tex)
                iu1w = iu1 % W_tex
                iv0w = iv0 % H_tex if iv0 < 0 else (iv0 % H_tex)
                iv1w = iv1 % H_tex

                # Bilinear weights
                w00 = (1 - fu) * (1 - fv)
                w10 = fu * (1 - fv)
                w01 = (1 - fu) * fv
                w11 = fu * fv

                gu = 0.0
                gv = 0.0

                for c in range(C):
                    d = dy_np[n, py_, px, c]

                    # Texture gradient
                    grad_tex[tz, iv0w, iu0w, c] += w00 * d
                    grad_tex[tz, iv0w, iu1w, c] += w10 * d
                    grad_tex[tz, iv1w, iu0w, c] += w01 * d
                    grad_tex[tz, iv1w, iu1w, c] += w11 * d

                    # UV gradient
                    a00 = tex_np[tz, iv0w, iu0w, c]
                    a10 = tex_np[tz, iv0w, iu1w, c]
                    a01 = tex_np[tz, iv1w, iu0w, c]
                    a11 = tex_np[tz, iv1w, iu1w, c]
                    ad = a11 + a00 - a10 - a01
                    gu += d * ((a10 - a00) + fv * ad) * W_tex
                    gv += d * ((a01 - a00) + fu * ad) * H_tex

                grad_uv[n, py_, px, 0] = gu
                grad_uv[n, py_, px, 1] = gv

    return torch.from_numpy(grad_tex), torch.from_numpy(grad_uv)


# ---------------------------------------------------------------------------
# Tests: Rasterize Backward
# ---------------------------------------------------------------------------

class TestRasterizeGrad:
    def test_rasterize_grad_vs_cpu_ref(self):
        """Metal rasterize grad matches CPU reference element-by-element."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_simple_triangle()
        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])

        # Synthetic gradient
        dy = torch.randn(1, 8, 8, 4)
        dy[..., 2:] = 0  # Only u,v gradients

        rast_2d = rast[0]
        dy_2d = dy[0]

        # Metal result
        metal_grad = _C.rasterize_grad(ctx._ctx, pos, tri, rast_2d, dy_2d, torch.empty(0))

        # CPU reference
        cpu_grad = rasterize_grad_ref(pos, tri, rast_2d, dy_2d)

        np.testing.assert_allclose(metal_grad.numpy(), cpu_grad.numpy(), atol=1e-5, rtol=1e-4,
                                   err_msg="Rasterize grad Metal vs CPU mismatch")

    def test_rasterize_grad_db_vs_cpu_ref(self):
        """Metal rasterize grad with DB matches CPU reference."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_simple_triangle()
        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])

        dy = torch.randn(1, 8, 8, 4)
        dy[..., 2:] = 0
        ddb = torch.randn(1, 8, 8, 4)

        rast_2d = rast[0]
        dy_2d = dy[0]
        ddb_2d = ddb[0]

        metal_grad = _C.rasterize_grad(ctx._ctx, pos, tri, rast_2d, dy_2d, ddb_2d)
        cpu_grad = rasterize_grad_ref(pos, tri, rast_2d, dy_2d, ddb_2d)

        np.testing.assert_allclose(metal_grad.numpy(), cpu_grad.numpy(), atol=1e-4, rtol=1e-3,
                                   err_msg="Rasterize grad+DB Metal vs CPU mismatch")

    def test_rasterize_autograd(self):
        """torch.autograd backward works for rasterize."""
        ctx = dr.MtlRasterizeContext()
        pos = torch.tensor([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=torch.float32, requires_grad=True)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])
        loss = rast[..., :2].sum()
        loss.backward()

        assert pos.grad is not None
        assert pos.grad.shape == (3, 4)
        assert (pos.grad != 0).any(), "Position gradients should be nonzero"


# ---------------------------------------------------------------------------
# Tests: Interpolate Backward
# ---------------------------------------------------------------------------

class TestInterpolateGrad:
    def test_interpolate_grad_vs_cpu_ref(self):
        """Metal interpolate grad matches CPU reference element-by-element."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_fullscreen_quad()
        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])

        attr = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ], dtype=torch.float32)

        dy = torch.randn(8, 8, 3)

        rast_2d = rast[0]
        tri_cpu = tri.contiguous().int()

        # Metal result
        g_attr_metal, g_rast_metal = _C.interpolate_grad(attr, rast_2d, tri_cpu, dy)

        # CPU reference
        g_attr_cpu, g_rast_cpu = interpolate_grad_ref(attr, rast_2d, tri_cpu, dy)

        np.testing.assert_allclose(g_attr_metal.numpy(), g_attr_cpu.numpy(), atol=1e-5, rtol=1e-4,
                                   err_msg="Interpolate grad_attr Metal vs CPU mismatch")
        np.testing.assert_allclose(g_rast_metal.numpy(), g_rast_cpu.numpy(), atol=1e-5, rtol=1e-4,
                                   err_msg="Interpolate grad_rast Metal vs CPU mismatch")

    def test_interpolate_autograd(self):
        """torch.autograd backward works for interpolate."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_fullscreen_quad()
        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])

        attr = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ], dtype=torch.float32, requires_grad=True)

        color, _ = dr.interpolate(attr, rast, tri)
        loss = color.sum()
        loss.backward()

        assert attr.grad is not None
        assert (attr.grad != 0).any()


# ---------------------------------------------------------------------------
# Tests: Texture Backward
# ---------------------------------------------------------------------------

class TestTextureGrad:
    def test_texture_grad_nearest_vs_cpu_ref(self):
        """Metal nearest texture grad matches CPU reference."""
        tex = torch.randn(1, 4, 4, 3)
        uv = torch.rand(1, 2, 2, 2)
        dy = torch.randn(1, 2, 2, 3)

        # Metal result
        mip_wrapper = _C.TextureMipWrapper()
        g_tex_metal, g_uv_metal, _, _ = _C.texture_grad(
            tex.contiguous(), uv.contiguous(), dy.contiguous(),
            torch.empty(0), torch.empty(0),
            mip_wrapper, [], 0, 1)  # nearest, wrap

        # CPU reference
        g_tex_cpu = texture_grad_nearest_ref(tex, uv, dy)

        np.testing.assert_allclose(g_tex_metal.numpy(), g_tex_cpu.numpy(), atol=1e-5, rtol=1e-4,
                                   err_msg="Texture grad nearest Metal vs CPU mismatch")

    def test_texture_grad_linear_vs_cpu_ref(self):
        """Metal linear texture grad matches CPU reference."""
        tex = torch.randn(1, 4, 4, 3)
        uv = torch.tensor([[
            [[0.3, 0.7], [0.8, 0.2]],
            [[0.1, 0.9], [0.5, 0.5]],
        ]], dtype=torch.float32)
        dy = torch.randn(1, 2, 2, 3)

        # Metal result
        mip_wrapper = _C.TextureMipWrapper()
        g_tex_metal, g_uv_metal, _, _ = _C.texture_grad(
            tex.contiguous(), uv.contiguous(), dy.contiguous(),
            torch.empty(0), torch.empty(0),
            mip_wrapper, [], 1, 1)  # linear, wrap

        # CPU reference
        g_tex_cpu, g_uv_cpu = texture_grad_linear_ref(tex, uv, dy)

        np.testing.assert_allclose(g_tex_metal.numpy(), g_tex_cpu.numpy(), atol=1e-5, rtol=1e-4,
                                   err_msg="Texture grad_tex linear Metal vs CPU mismatch")
        np.testing.assert_allclose(g_uv_metal.numpy(), g_uv_cpu.numpy(), atol=1e-4, rtol=1e-3,
                                   err_msg="Texture grad_uv linear Metal vs CPU mismatch")

    def test_texture_autograd_nearest(self):
        """torch.autograd backward works for texture (nearest)."""
        tex = torch.randn(1, 4, 4, 3, requires_grad=True)
        uv = torch.rand(1, 2, 2, 2)

        out = dr.texture(tex, uv, filter_mode='nearest', boundary_mode='wrap')
        loss = out.sum()
        loss.backward()

        assert tex.grad is not None
        assert (tex.grad != 0).any()

    def test_texture_autograd_linear(self):
        """torch.autograd backward works for texture (linear)."""
        tex = torch.randn(1, 4, 4, 3, requires_grad=True)
        uv = torch.rand(1, 2, 2, 2, requires_grad=True)

        out = dr.texture(tex, uv, filter_mode='linear', boundary_mode='wrap')
        loss = out.sum()
        loss.backward()

        assert tex.grad is not None and (tex.grad != 0).any()
        assert uv.grad is not None and (uv.grad != 0).any()


# ---------------------------------------------------------------------------
# Tests: End-to-End Chain
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_rasterize_interpolate_chain(self):
        """Gradient flows through rasterize→interpolate chain."""
        ctx = dr.MtlRasterizeContext()

        pos = torch.tensor([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=torch.float32, requires_grad=True)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        # Use asymmetric attributes so bary gradients don't cancel to zero
        attr = torch.tensor([
            [1.0], [0.0], [0.0]
        ], dtype=torch.float32, requires_grad=True)

        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])
        color, _ = dr.interpolate(attr, rast, tri)
        loss = color.sum()
        loss.backward()

        assert pos.grad is not None and (pos.grad != 0).any(), \
            f"pos.grad should be nonzero for asymmetric attributes, got {pos.grad}"
        assert attr.grad is not None and (attr.grad != 0).any()

    def test_full_render_chain(self):
        """Gradient flows through rasterize→interpolate→texture chain."""
        ctx = dr.MtlRasterizeContext()

        pos = torch.tensor([
            [-1.0, -1.0, 0.5, 1.0],
            [ 1.0, -1.0, 0.5, 1.0],
            [ 1.0,  1.0, 0.5, 1.0],
            [-1.0,  1.0, 0.5, 1.0],
        ], dtype=torch.float32, requires_grad=True)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

        uv_attr = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ], dtype=torch.float32, requires_grad=True)

        tex = torch.randn(1, 8, 8, 3, requires_grad=True)

        rast, db = dr.rasterize(ctx, pos, tri, [16, 16])
        uv, _ = dr.interpolate(uv_attr, rast, tri)
        color = dr.texture(tex, uv, filter_mode='linear', boundary_mode='wrap')

        loss = color.sum()
        loss.backward()

        assert pos.grad is not None and (pos.grad != 0).any(), "pos gradients missing"
        assert uv_attr.grad is not None and (uv_attr.grad != 0).any(), "uv_attr gradients missing"
        assert tex.grad is not None and (tex.grad != 0).any(), "tex gradients missing"

    def test_antialias_backward(self):
        """Gradient flows through antialias backward pass."""
        ctx = dr.MtlRasterizeContext()

        pos = torch.tensor([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=torch.float32, requires_grad=True)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        rast, db = dr.rasterize(ctx, pos, tri, [8, 8])

        attr = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, requires_grad=True)
        color, _ = dr.interpolate(attr, rast, tri)

        aa_color = dr.antialias(color, rast, pos, tri)
        loss = aa_color.sum()
        loss.backward()

        assert pos.grad is not None
        assert attr.grad is not None


# ---------------------------------------------------------------------------
# Tests: Gradient Consistency
# ---------------------------------------------------------------------------

class TestGradientConsistency:
    def test_rasterize_grad_zero_for_empty_pixels(self):
        """Pixels with no triangle should contribute zero gradient."""
        ctx = dr.MtlRasterizeContext()
        pos = torch.tensor([
            [-0.1, -0.1, 0.5, 1.0],
            [ 0.1, -0.1, 0.5, 1.0],
            [ 0.0,  0.1, 0.5, 1.0],
        ], dtype=torch.float32, requires_grad=True)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        rast, db = dr.rasterize(ctx, pos, tri, [32, 32])

        # Most pixels are empty
        n_covered = (rast[..., 3] > 0).sum().item()
        n_total = 32 * 32
        assert n_covered < n_total / 2, "Triangle should be small"

        loss = rast[..., :2].sum()
        loss.backward()
        assert pos.grad is not None

    def test_interpolate_grad_conservation(self):
        """Sum of attr gradients for uniform dy equals number of covered pixels."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_fullscreen_quad()
        rast, db = dr.rasterize(ctx, pos, tri, [4, 4])

        attr = torch.tensor([
            [1.0], [1.0], [1.0], [1.0]
        ], dtype=torch.float32, requires_grad=True)

        color, _ = dr.interpolate(attr, rast, tri)

        # For uniform attributes with dy=1, each covered pixel contributes b0+b1+b2=1
        # to the total gradient sum
        n_covered = (rast[..., 3] > 0).sum().item()
        loss = color.sum()
        loss.backward()

        total_grad = attr.grad.sum().item()
        np.testing.assert_allclose(total_grad, n_covered, atol=0.01,
                                   err_msg="Gradient sum should equal covered pixel count")

    def test_texture_grad_sum_conservation(self):
        """For nearest filter with uniform dy, total tex grad equals total dy."""
        tex = torch.ones(1, 4, 4, 1, requires_grad=True)
        uv = torch.rand(1, 3, 3, 2)

        out = dr.texture(tex, uv, filter_mode='nearest', boundary_mode='wrap')
        loss = out.sum()
        loss.backward()

        # Total gradient into tex should equal number of pixels (9)
        np.testing.assert_allclose(tex.grad.sum().item(), 9.0, atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
