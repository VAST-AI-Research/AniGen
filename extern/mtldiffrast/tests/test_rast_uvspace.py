"""
Tests for UV-space rasterization — the exact scenario that was broken.
When all z=0 (UV-space baking), EVERY triangle should get its correct face ID,
not just the first one in scan order.
"""
import torch
import numpy as np
import struct
import pytest


def float_to_triidx_ref(f):
    if f <= 16777216.0:
        return int(f)
    bits = struct.unpack('I', struct.pack('f', f))[0]
    return bits - 0x4a800000


class TestUVSpaceRasterization:
    """UV-space rasterization (all z=0) must produce correct face IDs."""

    @staticmethod
    def _get_mtl():
        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            return MtlRasterizeContext, rasterize
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_two_triangles_different_ids(self):
        """Two non-overlapping triangles at z=0: each region should have its own face ID."""
        MtlRasterizeContext, rasterize = self._get_mtl()

        # Left triangle and right triangle, both at z=0
        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],  # 0: bottom-left
            [ 0.0, -1.0, 0.0, 1.0],  # 1: bottom-center
            [-0.5,  1.0, 0.0, 1.0],  # 2: top-left
            [ 0.0, -1.0, 0.0, 1.0],  # 3: bottom-center
            [ 1.0, -1.0, 0.0, 1.0],  # 4: bottom-right
            [ 0.5,  1.0, 0.0, 1.0],  # 5: top-right
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)

        ctx = MtlRasterizeContext()
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[32, 32])
        mtl_rast = mtl_rast[0].numpy()

        # Collect face IDs
        face_ids = set()
        for py in range(32):
            for px in range(32):
                if mtl_rast[py, px, 3] != 0:
                    fid = float_to_triidx_ref(mtl_rast[py, px, 3]) - 1
                    face_ids.add(fid)

        # Both triangle 0 and triangle 1 must appear
        assert 0 in face_ids, "Triangle 0 not found in rasterization output"
        assert 1 in face_ids, "Triangle 1 not found in rasterization output"

    def test_uv_quad_face_ids(self):
        """UV-space quad (2 triangles covering full viewport) — both face IDs must appear."""
        MtlRasterizeContext, rasterize = self._get_mtl()

        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

        ctx = MtlRasterizeContext()
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[64, 64])
        mtl_rast = mtl_rast[0].numpy()

        face_ids = set()
        for py in range(64):
            for px in range(64):
                if mtl_rast[py, px, 3] != 0:
                    fid = float_to_triidx_ref(mtl_rast[py, px, 3]) - 1
                    face_ids.add(fid)

        assert face_ids == {0, 1}, f"Expected face IDs {{0, 1}}, got {face_ids}"

    def test_grid_mesh_all_faces(self):
        """Grid mesh with 8 triangles at z=0 — all 8 face IDs must appear."""
        MtlRasterizeContext, rasterize = self._get_mtl()

        # 3x3 grid of vertices = 2x2 grid of quads = 8 triangles
        verts = []
        for y in range(3):
            for x in range(3):
                # Map to [-1, 1] clip space
                cx = -1.0 + x * 1.0
                cy = -1.0 + y * 1.0
                verts.append([cx, cy, 0.0, 1.0])
        pos = torch.tensor(verts, dtype=torch.float32)

        tris = []
        for y in range(2):
            for x in range(2):
                i = y * 3 + x
                tris.append([i, i + 1, i + 4])
                tris.append([i, i + 4, i + 3])
        tri = torch.tensor(tris, dtype=torch.int32)

        ctx = MtlRasterizeContext()
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[64, 64])
        mtl_rast = mtl_rast[0].numpy()

        face_ids = set()
        for py in range(64):
            for px in range(64):
                if mtl_rast[py, px, 3] != 0:
                    fid = float_to_triidx_ref(mtl_rast[py, px, 3]) - 1
                    face_ids.add(fid)

        expected = set(range(8))
        assert face_ids == expected, f"Expected {expected}, got {face_ids}"

    def test_barycentrics_valid(self):
        """UV-space barycentrics should be valid (in [0,1], sum <= 1)."""
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

        mask = mtl_rast[:, :, 3] != 0
        u = mtl_rast[mask, 0]
        v = mtl_rast[mask, 1]

        assert (u >= -1e-6).all(), f"u has negative values: min={u.min()}"
        assert (v >= -1e-6).all(), f"v has negative values: min={v.min()}"
        assert ((u + v) <= 1.0 + 1e-5).all(), f"u+v exceeds 1: max={( u + v).max()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
