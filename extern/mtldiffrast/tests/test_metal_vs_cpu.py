"""
Absolute parity tests: Metal output vs CPU reference, element-by-element.

Every test reads Metal output back to numpy and compares against CPU
reference implementations of the differentiable rasterization math.
"""
import torch
import numpy as np
import struct
import math
import pytest

try:
    import mtldiffrast.torch as dr
    HAS_MTLDIFFRAST = True
except ImportError:
    HAS_MTLDIFFRAST = False

pytestmark = pytest.mark.skipif(not HAS_MTLDIFFRAST, reason="mtldiffrast not built")


# ============================================================
# CPU reference implementations
# ============================================================

def triidx_to_float_ref(x):
    if x <= 0x01000000:
        return float(x)
    bits = 0x4a800000 + x
    return struct.unpack('f', struct.pack('I', bits & 0xFFFFFFFF))[0]


def float_to_triidx_ref(f):
    if f <= 16777216.0:
        return int(f)
    bits = struct.unpack('I', struct.pack('f', f))[0]
    return bits - 0x4a800000


def rasterize_ref(pos, tri, H, W):
    """CPU rasterizer — edge function math for reference testing."""
    V = pos.shape[0]
    T = tri.shape[0]
    rast = np.zeros((H, W, 4), dtype=np.float32)
    rast_db = np.zeros((H, W, 4), dtype=np.float32)

    xs = 2.0 / W
    xo = -1.0 + 1.0 / W
    ys = 2.0 / H
    yo = -1.0 + 1.0 / H

    for py in range(H):
        for px in range(W):
            fx = xs * px + xo
            fy = ys * py + yo
            best_z = 2.0  # far sentinel; nvdiffrast keeps the smallest z/w
            for t in range(T):
                vi0, vi1, vi2 = int(tri[t, 0]), int(tri[t, 1]), int(tri[t, 2])
                if vi0 < 0 or vi0 >= V or vi1 < 0 or vi1 >= V or vi2 < 0 or vi2 >= V:
                    continue
                p0, p1, p2 = pos[vi0], pos[vi1], pos[vi2]

                p0x = p0[0] - fx * p0[3]
                p0y = p0[1] - fy * p0[3]
                p1x = p1[0] - fx * p1[3]
                p1y = p1[1] - fy * p1[3]
                p2x = p2[0] - fx * p2[3]
                p2y = p2[1] - fy * p2[3]
                a0 = p1x * p2y - p1y * p2x
                a1 = p2x * p0y - p2y * p0x
                a2 = p0x * p1y - p0y * p1x

                at = a0 + a1 + a2
                if at <= 0:
                    continue

                iw = 1.0 / at
                b0 = a0 * iw
                b1 = a1 * iw

                z = p0[2] * a0 + p1[2] * a1 + p2[2] * a2
                w = p0[3] * a0 + p1[3] * a1 + p2[3] * a2
                zw = z / w

                if b0 >= 0 and b1 >= 0 and (b0 + b1) <= 1.0 and zw < best_z:
                    dfxdx = xs * iw
                    dfydy = ys * iw
                    da0dx = p2[1] * p1[3] - p1[1] * p2[3]
                    da0dy = p1[0] * p2[3] - p2[0] * p1[3]
                    da1dx = p0[1] * p2[3] - p2[1] * p0[3]
                    da1dy = p2[0] * p0[3] - p0[0] * p2[3]
                    datdx = da0dx + da1dx + (p1[1] * p0[3] - p0[1] * p1[3])
                    datdy = da0dy + da1dy + (p0[0] * p1[3] - p1[0] * p0[3])
                    dudx = dfxdx * (b0 * datdx - da0dx)
                    dudy = dfydy * (b0 * datdy - da0dy)
                    dvdx = dfxdx * (b1 * datdx - da1dx)
                    dvdy = dfydy * (b1 * datdy - da1dy)

                    cb0 = max(0.0, min(1.0, b0))
                    cb1 = max(0.0, min(1.0, b1))
                    bs = 1.0 / max(cb0 + cb1, 1.0)
                    cb0 *= bs
                    cb1 *= bs
                    czw = max(-1.0, min(1.0, zw))

                    best_z = zw
                    tri_f = triidx_to_float_ref(t + 1)
                    rast[py, px] = [cb0, cb1, czw, tri_f]
                    rast_db[py, px] = [dudx, dudy, dvdx, dvdy]

    return rast, rast_db


def interpolate_ref(attr, rast, tri):
    """CPU interpolation — barycentric attribute weighting."""
    H, W, _ = rast.shape
    V, A = attr.shape
    T = tri.shape[0]
    out = np.zeros((H, W, A), dtype=np.float32)

    for py in range(H):
        for px in range(W):
            r = rast[py, px]
            tri_idx = float_to_triidx_ref(r[3]) - 1
            if tri_idx < 0 or tri_idx >= T:
                continue
            vi0 = int(tri[tri_idx, 0])
            vi1 = int(tri[tri_idx, 1])
            vi2 = int(tri[tri_idx, 2])
            b0, b1 = r[0], r[1]
            b2 = 1.0 - b0 - b1
            for i in range(A):
                out[py, px, i] = b0 * attr[vi0, i] + b1 * attr[vi1, i] + b2 * attr[vi2, i]

    return out


def texture_nearest_ref(tex, uv, H, W):
    """CPU nearest-neighbor texture sampling.
    tex: [TH, TW, C], uv: [H, W, 2] in [0,1] range.
    """
    TH, TW, C = tex.shape
    out = np.zeros((H, W, C), dtype=np.float32)
    for py in range(H):
        for px in range(W):
            u, v = uv[py, px, 0], uv[py, px, 1]
            # Wrap
            u = u - math.floor(u)
            v = v - math.floor(v)
            # Texel coords
            tx = min(int(u * TW), TW - 1)
            ty = min(int(v * TH), TH - 1)
            out[py, px] = tex[ty, tx]
    return out


def texture_linear_ref(tex, uv, H, W):
    """CPU bilinear texture sampling.
    tex: [TH, TW, C], uv: [H, W, 2] in [0,1] range.
    """
    TH, TW, C = tex.shape
    out = np.zeros((H, W, C), dtype=np.float32)
    for py in range(H):
        for px in range(W):
            u, v = uv[py, px, 0], uv[py, px, 1]
            u = u - math.floor(u)
            v = v - math.floor(v)
            # Texel centers
            fu = u * TW - 0.5
            fv = v * TH - 0.5
            tx0 = int(math.floor(fu))
            ty0 = int(math.floor(fv))
            tx1 = tx0 + 1
            ty1 = ty0 + 1
            fx = fu - tx0
            fy = fv - ty0
            # Wrap
            tx0 = tx0 % TW
            tx1 = tx1 % TW
            ty0 = ty0 % TH
            ty1 = ty1 % TH
            # Bilinear
            out[py, px] = (
                tex[ty0, tx0] * (1 - fx) * (1 - fy) +
                tex[ty0, tx1] * fx * (1 - fy) +
                tex[ty1, tx0] * (1 - fx) * fy +
                tex[ty1, tx1] * fx * fy
            )
    return out


# ============================================================
# RASTERIZE: element-by-element parity
# ============================================================

class TestRasterizeParity:

    @staticmethod
    def _compare(pos_np, tri_np, H, W):
        ref_rast, ref_db = rasterize_ref(pos_np, tri_np, H, W)
        pos = torch.from_numpy(pos_np)
        tri = torch.from_numpy(tri_np)
        ctx = dr.MtlRasterizeContext()
        mtl_rast, mtl_db = dr.rasterize(ctx, pos, tri, [H, W])
        mtl_rast = mtl_rast[0].numpy()
        mtl_db = mtl_db[0].numpy()

        mask = ref_rast[:, :, 3] != 0
        mtl_mask = mtl_rast[:, :, 3] != 0
        # Coverage: allow small edge-pixel differences from hardware rasterizer
        coverage_diff = np.abs(mask.astype(int) - mtl_mask.astype(int)).sum()
        assert coverage_diff <= max(2, int(0.05 * H * W)), \
            f"Coverage mismatch: {coverage_diff} pixels differ (H={H}, W={W})"

        # Compare only pixels covered by both
        both = mask & mtl_mask
        if both.any():
            # Barycentrics — hardware rasterizer has ~2e-4 precision vs software edge functions
            np.testing.assert_allclose(mtl_rast[both, 0], ref_rast[both, 0], atol=5e-4, rtol=1e-3)
            np.testing.assert_allclose(mtl_rast[both, 1], ref_rast[both, 1], atol=5e-4, rtol=1e-3)
            # Depth
            np.testing.assert_allclose(mtl_rast[both, 2], ref_rast[both, 2], atol=5e-4, rtol=1e-3)
            # Triangle IDs — exact for jointly covered pixels
            for py in range(H):
                for px in range(W):
                    if both[py, px]:
                        ref_id = float_to_triidx_ref(ref_rast[py, px, 3])
                        mtl_id = float_to_triidx_ref(mtl_rast[py, px, 3])
                        assert ref_id == mtl_id, f"Tri ID mismatch at ({px},{py})"
            # Bary derivatives
            np.testing.assert_allclose(mtl_db[both], ref_db[both], atol=5e-4, rtol=1e-3)

    def test_single_triangle_8x8(self):
        pos = np.array([[-0.5, -0.5, 0.5, 1], [0.5, -0.5, 0.5, 1], [0, 0.5, 0.5, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        self._compare(pos, tri, 8, 8)

    def test_single_triangle_16x16(self):
        pos = np.array([[-0.5, -0.5, 0.5, 1], [0.5, -0.5, 0.5, 1], [0, 0.5, 0.5, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        self._compare(pos, tri, 16, 16)

    def test_single_triangle_32x32(self):
        pos = np.array([[-0.5, -0.5, 0.5, 1], [0.5, -0.5, 0.5, 1], [0, 0.5, 0.5, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        self._compare(pos, tri, 32, 32)

    def test_fullscreen_quad_8x8(self):
        pos = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self._compare(pos, tri, 8, 8)

    def test_fullscreen_quad_32x32(self):
        pos = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self._compare(pos, tri, 32, 32)

    def test_depth_ordering(self):
        """Front triangle wins over back triangle — exact tri ID parity."""
        pos = np.array([
            [-0.8, -0.8, 0.3, 1], [0.8, -0.8, 0.3, 1], [0, 0.8, 0.3, 1],  # front
            [-0.5, -0.5, 0.7, 1], [0.5, -0.5, 0.7, 1], [0, 0.5, 0.7, 1],  # back
        ], dtype=np.float32)
        tri = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        self._compare(pos, tri, 16, 16)

    def test_non_square_resolution(self):
        pos = np.array([[-0.5, -0.5, 0.5, 1], [0.5, -0.5, 0.5, 1], [0, 0.5, 0.5, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        self._compare(pos, tri, 8, 16)
        self._compare(pos, tri, 16, 8)


# ============================================================
# INTERPOLATE: element-by-element parity
# ============================================================

class TestInterpolateParity:

    def test_position_interpolation(self):
        pos = np.array([[-0.5, -0.5, 0.5, 1], [0.5, -0.5, 0.5, 1], [0, 0.5, 0.5, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        attr = pos[:, :3]
        H, W = 16, 16

        ref_rast, _ = rasterize_ref(pos, tri, H, W)
        ref_interp = interpolate_ref(attr, ref_rast, tri)

        ctx = dr.MtlRasterizeContext()
        mtl_rast, mtl_db = dr.rasterize(ctx, torch.from_numpy(pos), torch.from_numpy(tri), [H, W])
        mtl_interp, _ = dr.interpolate(torch.from_numpy(attr), mtl_rast, torch.from_numpy(tri))
        mtl_interp = mtl_interp[0].numpy()

        mask = ref_rast[:, :, 3] != 0
        if mask.any():
            np.testing.assert_allclose(mtl_interp[mask], ref_interp[mask], atol=1e-5, rtol=1e-4)

    def test_uv_interpolation_fullscreen(self):
        pos = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        attr = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        H, W = 32, 32

        ref_rast, _ = rasterize_ref(pos, tri, H, W)
        ref_interp = interpolate_ref(attr, ref_rast, tri)

        ctx = dr.MtlRasterizeContext()
        mtl_rast, _ = dr.rasterize(ctx, torch.from_numpy(pos), torch.from_numpy(tri), [H, W])
        mtl_interp, _ = dr.interpolate(torch.from_numpy(attr), mtl_rast, torch.from_numpy(tri))
        mtl_interp = mtl_interp[0].numpy()

        mask = ref_rast[:, :, 3] != 0
        assert mask.all(), "Fullscreen quad should cover all pixels"
        np.testing.assert_allclose(mtl_interp, ref_interp, atol=1e-5, rtol=1e-4)

    def test_9_attributes(self):
        """pos(3) + normal(3) + color(3)."""
        pos = np.array([[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2]], dtype=np.int32)
        attr = np.array([
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0],
            [.5, 1, 0, 0, 0, 1, 0, 0, 1],
        ], dtype=np.float32)
        H, W = 8, 8

        ref_rast, _ = rasterize_ref(pos, tri, H, W)
        ref_interp = interpolate_ref(attr, ref_rast, tri)

        ctx = dr.MtlRasterizeContext()
        mtl_rast, _ = dr.rasterize(ctx, torch.from_numpy(pos), torch.from_numpy(tri), [H, W])
        mtl_interp, _ = dr.interpolate(torch.from_numpy(attr), mtl_rast, torch.from_numpy(tri))
        mtl_interp = mtl_interp[0].numpy()

        mask = ref_rast[:, :, 3] != 0
        if mask.any():
            np.testing.assert_allclose(mtl_interp[mask], ref_interp[mask], atol=1e-5, rtol=1e-4)


# ============================================================
# TEXTURE: element-by-element parity
# ============================================================

class TestTextureNearestParity:
    """Nearest-neighbor texture: Metal vs CPU reference, pixel-exact."""

    def test_checkerboard_4x4(self):
        """4x4 checkerboard sampled at 4x4 pixels — each pixel hits one texel."""
        ctx = dr.MtlRasterizeContext()
        pos = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        H, W = 4, 4

        rast, _ = dr.rasterize(ctx, torch.from_numpy(pos), torch.from_numpy(tri), [H, W])

        tex_np = np.zeros((4, 4, 3), dtype=np.float32)
        for y in range(4):
            for x in range(4):
                tex_np[y, x] = [1, 0, 0] if (x + y) % 2 == 0 else [0, 1, 0]
        tex = torch.from_numpy(tex_np).unsqueeze(0)  # [1, 4, 4, 3]

        uv = rast[..., :2]  # barycentrics as UVs
        mtl_out = dr.texture(tex, uv, filter_mode='nearest')
        mtl_np = mtl_out[0].numpy()

        # CPU ref
        uv_np = uv[0].numpy()
        ref_out = texture_nearest_ref(tex_np, uv_np, H, W)

        # Compare only covered pixels
        mask = rast[0, :, :, 3].numpy() != 0
        if mask.any():
            np.testing.assert_allclose(mtl_np[mask], ref_out[mask], atol=1e-6)


class TestTextureLinearParity:
    """Bilinear texture: Metal vs CPU reference."""

    def test_gradient_4x4(self):
        ctx = dr.MtlRasterizeContext()
        pos = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float32)
        tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        H, W = 8, 8

        rast, _ = dr.rasterize(ctx, torch.from_numpy(pos), torch.from_numpy(tri), [H, W])

        tex_np = np.zeros((4, 4, 1), dtype=np.float32)
        for x in range(4):
            tex_np[:, x, 0] = x / 3.0
        tex = torch.from_numpy(tex_np).unsqueeze(0)

        uv = rast[..., :2]
        mtl_out = dr.texture(tex, uv, filter_mode='linear')
        mtl_np = mtl_out[0].numpy()

        uv_np = uv[0].numpy()
        ref_out = texture_linear_ref(tex_np, uv_np, H, W)

        mask = rast[0, :, :, 3].numpy() != 0
        if mask.any():
            np.testing.assert_allclose(mtl_np[mask], ref_out[mask], atol=1e-4, rtol=1e-3)


# ============================================================
# FULL PIPELINE: rasterize → interpolate → texture
# ============================================================

class TestFullRenderPipeline:
    """End-to-end: rasterize → interpolate UVs → texture lookup."""

    def test_textured_quad(self):
        """Rasterize quad, interpolate UVs, sample texture — full chain parity."""
        pos_np = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float32)
        tri_np = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uv_attr_np = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        H, W = 16, 16

        # CPU reference pipeline
        ref_rast, _ = rasterize_ref(pos_np, tri_np, H, W)
        ref_uv = interpolate_ref(uv_attr_np, ref_rast, tri_np)  # [H, W, 2]

        tex_np = np.zeros((8, 8, 3), dtype=np.float32)
        for y in range(8):
            for x in range(8):
                tex_np[y, x] = [x / 7.0, y / 7.0, 0.5]
        ref_tex = texture_linear_ref(tex_np, ref_uv, H, W)

        # Metal pipeline
        ctx = dr.MtlRasterizeContext()
        pos = torch.from_numpy(pos_np)
        tri = torch.from_numpy(tri_np)
        mtl_rast, _ = dr.rasterize(ctx, pos, tri, [H, W])
        mtl_uv, _ = dr.interpolate(torch.from_numpy(uv_attr_np), mtl_rast, tri)

        tex = torch.from_numpy(tex_np).unsqueeze(0)
        mtl_tex = dr.texture(tex, mtl_uv, filter_mode='linear')
        mtl_tex_np = mtl_tex[0].numpy()

        mask = ref_rast[:, :, 3] != 0
        assert mask.all(), "Fullscreen quad"
        # Pipeline error accumulates across 3 stages, so relax tolerance slightly
        np.testing.assert_allclose(mtl_tex_np[mask], ref_tex[mask], atol=1e-3, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
