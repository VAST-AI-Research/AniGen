"""
Tests for hardware render pipeline vs compute kernel parity.
Verifies that the Metal render pipeline produces identical results to the
compute kernel fallback for various geometries.
"""
import torch
import numpy as np
import struct
import pytest


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
    """CPU reference rasterizer matching nvdiffrast: smallest z/w wins, ties keep-first."""
    V = pos.shape[0]
    T = tri.shape[0]
    rast = np.zeros((H, W, 4), dtype=np.float32)

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
                    cb0 = max(0.0, min(1.0, b0))
                    cb1 = max(0.0, min(1.0, b1))
                    bs = 1.0 / max(cb0 + cb1, 1.0)
                    cb0 *= bs
                    cb1 *= bs
                    czw = max(-1.0, min(1.0, zw))
                    best_z = zw
                    tri_f = triidx_to_float_ref(t + 1)
                    rast[py, px] = [cb0, cb1, czw, tri_f]
    return rast


class TestRenderPipeline:
    """Hardware render pipeline should produce results matching compute kernel."""

    @staticmethod
    def _get_mtl():
        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            return MtlRasterizeContext, rasterize
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_single_triangle(self):
        MtlRasterizeContext, rasterize = self._get_mtl()
        pos = torch.tensor([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        ctx = MtlRasterizeContext()
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[16, 16])
        mtl_rast = mtl_rast[0].numpy()
        ref = rasterize_ref(pos.numpy(), tri.numpy(), 16, 16)

        mask = ref[:, :, 3] != 0
        if mask.any():
            np.testing.assert_allclose(mtl_rast[mask, :3], ref[mask, :3], atol=1e-4, rtol=1e-3)
            # Tri IDs must match exactly
            for py in range(16):
                for px in range(16):
                    if ref[py, px, 3] != 0:
                        assert float_to_triidx_ref(mtl_rast[py, px, 3]) == float_to_triidx_ref(ref[py, px, 3])

    def test_depth_test_two_triangles(self):
        """Closer triangle (smaller z/w) should win — nvdiffrast convention."""
        MtlRasterizeContext, rasterize = self._get_mtl()
        pos = torch.tensor([
            [-0.8, -0.8, 0.3, 1.0],
            [ 0.8, -0.8, 0.3, 1.0],
            [ 0.0,  0.8, 0.3, 1.0],
            [-0.5, -0.5, 0.7, 1.0],
            [ 0.5, -0.5, 0.7, 1.0],
            [ 0.0,  0.5, 0.7, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)

        ctx = MtlRasterizeContext()
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[16, 16])
        mtl_rast = mtl_rast[0].numpy()
        ref = rasterize_ref(pos.numpy(), tri.numpy(), 16, 16)

        mask = ref[:, :, 3] != 0
        if mask.any():
            for py in range(16):
                for px in range(16):
                    if ref[py, px, 3] != 0:
                        ref_id = float_to_triidx_ref(ref[py, px, 3])
                        mtl_id = float_to_triidx_ref(mtl_rast[py, px, 3])
                        assert ref_id == mtl_id, f"Depth test failed at ({px},{py}): ref={ref_id} mtl={mtl_id}"

    def test_fullscreen_quad_coverage(self):
        """Full-screen quad should cover every pixel."""
        MtlRasterizeContext, rasterize = self._get_mtl()
        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

        ctx = MtlRasterizeContext()
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[32, 32])
        mtl_rast = mtl_rast[0].numpy()

        covered = (mtl_rast[:, :, 3] != 0).sum()
        assert covered == 32 * 32, f"Expected full coverage, got {covered}/{32*32}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
