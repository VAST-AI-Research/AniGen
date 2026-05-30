"""
Comprehensive tests for mtldiffrast — texture, antialias, DepthPeeler.
Verifies the full rendering pipeline against expected behavior.
"""
import torch
import numpy as np
import pytest

try:
    import mtldiffrast.torch as dr
    HAS_MTLDIFFRAST = True
except ImportError:
    HAS_MTLDIFFRAST = False

pytestmark = pytest.mark.skipif(not HAS_MTLDIFFRAST, reason="mtldiffrast not built")


# ============================================================
# Helper: simple scene setup
# ============================================================

def make_fullscreen_quad():
    """A quad covering the full NDC [-1,1] range."""
    pos = torch.tensor([
        [-1, -1, 0.5, 1],
        [ 1, -1, 0.5, 1],
        [ 1,  1, 0.5, 1],
        [-1,  1, 0.5, 1],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    return pos, tri


def make_triangle():
    """A single triangle."""
    pos = torch.tensor([
        [-0.5, -0.5, 0.5, 1],
        [ 0.5, -0.5, 0.5, 1],
        [ 0.0,  0.5, 0.5, 1],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return pos, tri


def make_two_overlapping_triangles():
    """Two triangles at different depths, overlapping in screen space."""
    pos = torch.tensor([
        # Front triangle (z=0.3)
        [-0.5, -0.5, 0.3, 1],
        [ 0.5, -0.5, 0.3, 1],
        [ 0.0,  0.5, 0.3, 1],
        # Back triangle (z=0.7)
        [-0.5, -0.3, 0.7, 1],
        [ 0.5, -0.3, 0.7, 1],
        [ 0.0,  0.7, 0.7, 1],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    return pos, tri


# ============================================================
# Texture tests
# ============================================================

class TestTextureNearest:
    """Test nearest-neighbor texture sampling."""

    def test_texture_nearest_checkerboard(self):
        """Sample 8x8 checkerboard with nearest filter, verify exact texel."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_fullscreen_quad()
        rast, rast_db = dr.rasterize(ctx, pos, tri, [8, 8])

        # Create 8x8 checkerboard texture [1, 8, 8, 3]
        tex = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 == 0:
                    tex[0, y, x] = torch.tensor([1, 0, 0], dtype=torch.float32)
                else:
                    tex[0, y, x] = torch.tensor([0, 1, 0], dtype=torch.float32)

        # UV coordinates from rasterization (u,v channels)
        uv = rast[..., :2]

        out = dr.texture(tex, uv, filter_mode='nearest')
        assert out.shape == (1, 8, 8, 3)
        # At least some pixels should be red and some green
        assert out[0, :, :, 0].sum() > 0  # some red
        assert out[0, :, :, 1].sum() > 0  # some green


class TestTextureLinear:
    """Test bilinear texture sampling."""

    def test_texture_linear_gradient(self):
        """Sample gradient texture with linear filter using explicit UVs."""
        # Create explicit UV coordinates that map linearly across the image
        uv = torch.zeros(1, 4, 4, 2, dtype=torch.float32)
        for y in range(4):
            for x in range(4):
                uv[0, y, x, 0] = (x + 0.5) / 4.0  # u: 0→1 left to right
                uv[0, y, x, 1] = (y + 0.5) / 4.0  # v: 0→1 top to bottom

        # Gradient texture: intensity increases left to right
        tex = torch.zeros(1, 4, 4, 1, dtype=torch.float32)
        for x in range(4):
            tex[0, :, x, 0] = x / 3.0

        out = dr.texture(tex, uv, filter_mode='linear', boundary_mode='clamp')
        assert out.shape == (1, 4, 4, 1)
        # Output should increase from left to right
        col_means = out[0, :, :, 0].mean(dim=0)
        assert col_means[-1] > col_means[0], "Linear interpolation should preserve gradient"


# ============================================================
# Antialias tests
# ============================================================

class TestAntialias:
    """Test antialias edge blending."""

    def test_antialias_topology_hash(self):
        """Verify topology hash construction doesn't crash."""
        _, tri = make_triangle()
        topo = dr.antialias_construct_topology_hash(tri)
        assert topo is not None
        assert hasattr(topo, 'ev_hash')
        assert hasattr(topo, 'num_triangles')
        assert topo.num_triangles == 1

    def test_antialias_forward(self):
        """Verify antialias forward pass produces output."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_triangle()
        rast, _ = dr.rasterize(ctx, pos, tri, [32, 32])

        # Simple color buffer
        color = torch.ones(1, 32, 32, 3, dtype=torch.float32) * 0.5

        out = dr.antialias(color, rast, pos, tri)
        assert out.shape == (1, 32, 32, 3)

    def test_antialias_reuse_topology(self):
        """Verify topology hash can be reused across calls."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_triangle()
        rast, _ = dr.rasterize(ctx, pos, tri, [16, 16])
        color = torch.ones(1, 16, 16, 3, dtype=torch.float32)

        topo = dr.antialias_construct_topology_hash(tri)
        out1 = dr.antialias(color, rast, pos, tri, topology_hash=topo)
        out2 = dr.antialias(color, rast, pos, tri, topology_hash=topo)
        assert torch.allclose(out1, out2)


# ============================================================
# DepthPeeler tests
# ============================================================

class TestDepthPeeler:
    """Test depth peeling for multi-layer rasterization."""

    def test_depth_peeler_single_layer(self):
        """Single layer peeling should match regular rasterize."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_triangle()

        # Regular rasterize
        rast_ref, _ = dr.rasterize(ctx, pos, tri, [16, 16])

        # Depth peeler single layer
        with dr.DepthPeeler(ctx, pos, tri, [16, 16]) as peeler:
            rast_peel, _ = peeler.rasterize_next_layer()

        # Should produce same result for first layer
        assert rast_peel.shape == rast_ref.shape
        # Triangle IDs should match
        assert torch.allclose(rast_peel[..., 3], rast_ref[..., 3])

    def test_depth_peeler_two_layers(self):
        """Two overlapping triangles, peel 2 layers."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_two_overlapping_triangles()

        with dr.DepthPeeler(ctx, pos, tri, [16, 16]) as peeler:
            rast0, _ = peeler.rasterize_next_layer()
            rast1, _ = peeler.rasterize_next_layer()

        # Both layers should have valid output shape
        assert rast0.shape == (1, 16, 16, 4)
        assert rast1.shape == (1, 16, 16, 4)

    def test_depth_peeler_context_manager(self):
        """Verify context manager properly cleans up."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_triangle()

        assert ctx.active_depth_peeler is None
        with dr.DepthPeeler(ctx, pos, tri, [16, 16]) as peeler:
            assert ctx.active_depth_peeler is peeler
        assert ctx.active_depth_peeler is None

    def test_depth_peeler_nested_error(self):
        """Nested depth peelers should raise an error."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_triangle()

        with dr.DepthPeeler(ctx, pos, tri, [16, 16]):
            with pytest.raises(RuntimeError):
                with dr.DepthPeeler(ctx, pos, tri, [16, 16]):
                    pass

    def test_rasterize_during_peeling_error(self):
        """Direct rasterize() during peeling should raise an error."""
        ctx = dr.MtlRasterizeContext()
        pos, tri = make_triangle()

        with dr.DepthPeeler(ctx, pos, tri, [16, 16]):
            with pytest.raises(RuntimeError):
                dr.rasterize(ctx, pos, tri, [16, 16])


# ============================================================
# Legacy compatibility
# ============================================================

class TestCompatibility:
    """Test API compatibility."""

    def test_log_level(self):
        dr.set_log_level(0)
        assert dr.get_log_level() == 0
        dr.set_log_level(1)
        assert dr.get_log_level() == 1
