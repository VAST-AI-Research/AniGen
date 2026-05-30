"""
Tests for chunked rasterization (>100K triangles) with correct global face IDs.
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


class TestChunkedRasterization:
    """Large meshes that exceed the 100K chunk size should produce correct global face IDs."""

    @staticmethod
    def _get_mtl():
        try:
            from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
            return MtlRasterizeContext, rasterize
        except ImportError:
            pytest.skip("mtldiffrast not built")

    def test_face_ids_beyond_chunk_boundary(self):
        """Generate >100K triangles; face IDs in later chunks must use global offsets."""
        MtlRasterizeContext, rasterize = self._get_mtl()

        # Create a grid mesh with enough triangles to exceed 100K
        # 250x250 grid = 62500 quads = 125000 triangles
        grid_n = 250
        verts = []
        for y in range(grid_n + 1):
            for x in range(grid_n + 1):
                cx = -1.0 + 2.0 * x / grid_n
                cy = -1.0 + 2.0 * y / grid_n
                verts.append([cx, cy, 0.5, 1.0])

        tris = []
        for y in range(grid_n):
            for x in range(grid_n):
                i = y * (grid_n + 1) + x
                tris.append([i, i + 1, i + grid_n + 2])
                tris.append([i, i + grid_n + 2, i + grid_n + 1])

        pos = torch.tensor(verts, dtype=torch.float32)
        tri = torch.tensor(tris, dtype=torch.int32)

        assert tri.shape[0] == 125000, f"Expected 125000 triangles, got {tri.shape[0]}"

        ctx = MtlRasterizeContext()
        H, W = 64, 64
        mtl_rast, _ = rasterize(ctx, pos, tri, resolution=[H, W])
        mtl_rast = mtl_rast[0].numpy()

        # Every pixel should be covered
        covered = (mtl_rast[:, :, 3] != 0).sum()
        assert covered == H * W, f"Expected full coverage, got {covered}/{H*W}"

        # Collect face IDs — should include IDs > 100000 (beyond first chunk)
        max_fid = -1
        for py in range(H):
            for px in range(W):
                if mtl_rast[py, px, 3] != 0:
                    fid = float_to_triidx_ref(mtl_rast[py, px, 3]) - 1
                    assert fid >= 0, f"Invalid face ID {fid} at ({px},{py})"
                    assert fid < 125000, f"Face ID {fid} out of range at ({px},{py})"
                    max_fid = max(max_fid, fid)

        # With 125K triangles on a 64x64 grid, we must see face IDs > 100K
        # (the bottom-right corner maps to the later triangles)
        assert max_fid > 100000, f"Max face ID {max_fid} — expected > 100000 (chunking offset bug)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
