"""
Parity tests for mtldiffrast Metal implementation.
Tests rasterize + interpolate against CPU reference implementations
(edge functions, barycentrics, tri ID encoding, pixel differentials).
"""
import torch
import numpy as np
import math
import struct
import pytest


# ============================================================
# CPU reference implementations
# ============================================================

def triidx_to_float_ref(x):
    """Triangle ID encoding: int -> float."""
    if x <= 0x01000000:
        return float(x)
    # Biased bitwise mapping for large IDs
    bits = 0x4a800000 + x
    return struct.unpack('f', struct.pack('I', bits & 0xFFFFFFFF))[0]

def float_to_triidx_ref(f):
    """Triangle ID decoding: float -> int."""
    if f <= 16777216.0:
        return int(f)
    bits = struct.unpack('I', struct.pack('f', f))[0]
    return bits - 0x4a800000

def rasterize_ref(pos, tri, H, W):
    """CPU reference rasterizer — edge function evaluation per pixel.

    Depth convention matches nvdiffrast's cudaraster: the SMALLEST z/w wins
    (atomicMin), and equal-z/w fragments keep the first (lowest-index) triangle.
    """
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
                p0 = pos[vi0]
                p1 = pos[vi1]
                p2 = pos[vi2]

                # Edge functions
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
                    # Bary pixel differentials (pre-clamp b0/b1)
                    dfxdx = xs * iw
                    dfydy = ys * iw
                    da0dx = p2[1]*p1[3] - p1[1]*p2[3]
                    da0dy = p1[0]*p2[3] - p2[0]*p1[3]
                    da1dx = p0[1]*p2[3] - p2[1]*p0[3]
                    da1dy = p2[0]*p0[3] - p0[0]*p2[3]
                    da2dx = p1[1]*p0[3] - p0[1]*p1[3]
                    da2dy = p0[0]*p1[3] - p1[0]*p0[3]
                    datdx = da0dx + da1dx + da2dx
                    datdy = da0dy + da1dy + da2dy
                    dudx = dfxdx * (b0 * datdx - da0dx)
                    dudy = dfydy * (b0 * datdy - da0dy)
                    dvdx = dfxdx * (b1 * datdx - da1dx)
                    dvdy = dfydy * (b1 * datdy - da1dy)

                    # Clamp (after computing differentials)
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
    """CPU reference interpolation — barycentric attribute weighting."""
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
            if vi0 < 0 or vi0 >= V or vi1 < 0 or vi1 >= V or vi2 < 0 or vi2 >= V:
                continue
            b0 = r[0]
            b1 = r[1]
            b2 = 1.0 - b0 - b1
            for i in range(A):
                out[py, px, i] = b0 * attr[vi0, i] + b1 * attr[vi1, i] + b2 * attr[vi2, i]

    return out


# ============================================================
# Tests
# ============================================================

class TestTriIdxEncoding:
    """Test triangle ID float encoding roundtrips exactly."""

    def test_small_ids(self):
        for x in [0, 1, 2, 100, 1000, 16777216]:
            f = triidx_to_float_ref(x)
            assert float_to_triidx_ref(f) == x, f"Roundtrip failed for {x}"

    def test_large_ids(self):
        for x in [16777217, 50000000, 100000000, 889192447]:
            f = triidx_to_float_ref(x)
            assert float_to_triidx_ref(f) == x, f"Roundtrip failed for {x}"

    def test_sequential(self):
        """All IDs from 0 to 10000 roundtrip correctly."""
        for x in range(10001):
            f = triidx_to_float_ref(x)
            assert float_to_triidx_ref(f) == x


class TestRasterizeParity:
    """Test Metal rasterize output matches CPU reference."""

    @staticmethod
    def make_single_triangle():
        """A single triangle covering part of the viewport."""
        pos = torch.tensor([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        return pos, tri

    @staticmethod
    def make_two_triangles():
        """Two triangles, one in front of the other (depth test)."""
        pos = torch.tensor([
            [-0.8, -0.8, 0.3, 1.0],
            [ 0.8, -0.8, 0.3, 1.0],
            [ 0.0,  0.8, 0.3, 1.0],
            [-0.5, -0.5, 0.7, 1.0],
            [ 0.5, -0.5, 0.7, 1.0],
            [ 0.0,  0.5, 0.7, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        return pos, tri

    @staticmethod
    def make_uv_quad():
        """UV-space quad for texture baking (TRELLIS.2 use case)."""
        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
        return pos, tri

    def test_single_triangle_rast(self):
        pos, tri = self.make_single_triangle()
        H, W = 16, 16
        ref_rast, ref_db = rasterize_ref(pos.numpy(), tri.numpy(), H, W)

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            ctx = MtlRasterizeContext()
            mtl_rast, mtl_db = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_rast = mtl_rast[0].numpy()
            mtl_db = mtl_db[0].numpy()

            # Compare barycentrics (u, v) — skip empty pixels
            mask = ref_rast[:, :, 3] != 0
            if mask.any():
                np.testing.assert_allclose(mtl_rast[mask, 0], ref_rast[mask, 0], atol=1e-5, rtol=1e-4)
                np.testing.assert_allclose(mtl_rast[mask, 1], ref_rast[mask, 1], atol=1e-5, rtol=1e-4)
                np.testing.assert_allclose(mtl_rast[mask, 2], ref_rast[mask, 2], atol=1e-5, rtol=1e-4)

                # Triangle IDs must match exactly
                for py in range(H):
                    for px in range(W):
                        if ref_rast[py, px, 3] != 0:
                            ref_id = float_to_triidx_ref(ref_rast[py, px, 3])
                            mtl_id = float_to_triidx_ref(mtl_rast[py, px, 3])
                            assert ref_id == mtl_id, f"Tri ID mismatch at ({px},{py}): ref={ref_id} mtl={mtl_id}"

                # Bary differentials
                np.testing.assert_allclose(mtl_db[mask], ref_db[mask], atol=1e-4, rtol=1e-3)
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_two_triangles_depth(self):
        pos, tri = self.make_two_triangles()
        H, W = 16, 16
        ref_rast, ref_db = rasterize_ref(pos.numpy(), tri.numpy(), H, W)

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            ctx = MtlRasterizeContext()
            mtl_rast, mtl_db = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_rast = mtl_rast[0].numpy()

            mask = ref_rast[:, :, 3] != 0
            if mask.any():
                for py in range(H):
                    for px in range(W):
                        if ref_rast[py, px, 3] != 0:
                            ref_id = float_to_triidx_ref(ref_rast[py, px, 3])
                            mtl_id = float_to_triidx_ref(mtl_rast[py, px, 3])
                            assert ref_id == mtl_id, f"Depth test failed at ({px},{py})"
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_uv_quad_coverage(self):
        pos, tri = self.make_uv_quad()
        H, W = 32, 32
        ref_rast, _ = rasterize_ref(pos.numpy(), tri.numpy(), H, W)

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            ctx = MtlRasterizeContext()
            mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_rast = mtl_rast[0].numpy()

            # Every pixel should be covered
            ref_covered = (ref_rast[:, :, 3] != 0).sum()
            mtl_covered = (mtl_rast[:, :, 3] != 0).sum()
            assert ref_covered == mtl_covered, f"Coverage mismatch: ref={ref_covered} mtl={mtl_covered}"
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_empty_pixels(self):
        pos, tri = self.make_single_triangle()
        H, W = 8, 8
        ref_rast, ref_db = rasterize_ref(pos.numpy(), tri.numpy(), H, W)

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            ctx = MtlRasterizeContext()
            mtl_rast, mtl_db = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_rast = mtl_rast[0].numpy()
            mtl_db = mtl_db[0].numpy()

            # Empty pixels must be exactly zero
            empty = ref_rast[:, :, 3] == 0
            np.testing.assert_equal(mtl_rast[empty], 0.0)
            np.testing.assert_equal(mtl_db[empty], 0.0)
        except ImportError:
            pytest.skip("mtldiffrast not built")


class TestInterpolateParity:
    """Test Metal interpolate output matches CPU reference."""

    def test_position_interpolation(self):
        """Interpolate vertex positions — should reproduce positions at pixel centers."""
        pos = torch.tensor([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        attr = pos[:, :3]  # xyz positions as attributes
        H, W = 16, 16

        ref_rast, _ = rasterize_ref(pos.numpy(), tri.numpy(), H, W)
        ref_interp = interpolate_ref(attr.numpy(), ref_rast, tri.numpy())

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize, interpolate
            ctx = MtlRasterizeContext()
            mtl_rast, mtl_db = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_interp, _ = interpolate(attr, mtl_rast, tri)
            mtl_interp = mtl_interp[0].numpy()

            mask = ref_rast[:, :, 3] != 0
            if mask.any():
                np.testing.assert_allclose(mtl_interp[mask], ref_interp[mask], atol=1e-5, rtol=1e-4)
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_uv_interpolation(self):
        """Interpolate UV coordinates for texture baking."""
        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
        attr = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        H, W = 32, 32

        ref_rast, _ = rasterize_ref(pos.numpy(), tri.numpy(), H, W)
        ref_interp = interpolate_ref(attr.numpy(), ref_rast, tri.numpy())

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize, interpolate
            ctx = MtlRasterizeContext()
            mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_interp, _ = interpolate(attr, mtl_rast, tri)
            mtl_interp = mtl_interp[0].numpy()

            mask = ref_rast[:, :, 3] != 0
            if mask.any():
                np.testing.assert_allclose(mtl_interp[mask], ref_interp[mask], atol=1e-5, rtol=1e-4)
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_multi_attribute(self):
        """Interpolate multiple attributes (position + normal + color)."""
        pos = torch.tensor([
            [-0.5, -0.5, 0.0, 1.0],
            [ 0.5, -0.5, 0.0, 1.0],
            [ 0.0,  0.5, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        # 9 attributes: pos(3) + normal(3) + color(3)
        attr = torch.tensor([
            [0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        H, W = 8, 8

        ref_rast, _ = rasterize_ref(pos.numpy(), tri.numpy(), H, W)
        ref_interp = interpolate_ref(attr.numpy(), ref_rast, tri.numpy())

        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize, interpolate
            ctx = MtlRasterizeContext()
            mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[H, W])
            mtl_interp, _ = interpolate(attr, mtl_rast, tri)
            mtl_interp = mtl_interp[0].numpy()

            mask = ref_rast[:, :, 3] != 0
            if mask.any():
                np.testing.assert_allclose(mtl_interp[mask], ref_interp[mask], atol=1e-5, rtol=1e-4)
        except ImportError:
            pytest.skip("mtldiffrast not built")


class TestReferenceOnly:
    """Tests that run without the Metal extension — validate the reference impls."""

    def test_ref_single_triangle(self):
        pos = np.array([[-0.5,-0.5,0.5,1], [0.5,-0.5,0.5,1], [0,0.5,0.5,1]], dtype=np.float32)
        tri = np.array([[0,1,2]], dtype=np.int32)
        rast, db = rasterize_ref(pos, tri, 8, 8)
        # Center pixels should have a triangle
        assert rast[4, 4, 3] != 0, "Center pixel should have triangle"
        # Corner pixels likely empty
        assert rast[0, 0, 3] == 0, "Top-left corner should be empty"
        # Barycentrics should sum to <= 1
        mask = rast[:,:,3] != 0
        bsum = rast[mask, 0] + rast[mask, 1]
        assert (bsum <= 1.0 + 1e-6).all()

    def test_ref_interpolate_identity(self):
        """If attr = vertex positions and barycentrics are correct,
        interpolated positions should lie on the triangle plane."""
        pos = np.array([[-1,-1,0,1], [1,-1,0,1], [0,1,0,1]], dtype=np.float32)
        tri = np.array([[0,1,2]], dtype=np.int32)
        attr = pos[:, :3]
        rast, _ = rasterize_ref(pos, tri, 16, 16)
        interp = interpolate_ref(attr, rast, tri)
        mask = rast[:,:,3] != 0
        # All z coordinates should be 0 (triangle on z=0 plane)
        np.testing.assert_allclose(interp[mask, 2], 0.0, atol=1e-6)

    def test_ref_triidx_roundtrip(self):
        for x in range(1, 1000):
            assert float_to_triidx_ref(triidx_to_float_ref(x)) == x

    def test_ref_fullscreen_quad(self):
        """Full-screen quad should cover every pixel."""
        pos = np.array([[-1,-1,0,1],[1,-1,0,1],[1,1,0,1],[-1,1,0,1]], dtype=np.float32)
        tri = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        rast, _ = rasterize_ref(pos, tri, 8, 8)
        assert (rast[:,:,3] != 0).all(), "Full-screen quad should cover all pixels"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
