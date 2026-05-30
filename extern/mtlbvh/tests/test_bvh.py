"""
Tests for mtlbvh — Metal BVH distance queries and ray tracing.

Verifies unsigned/signed distance and ray tracing against brute-force
CPU reference implementations.
"""
import torch
import numpy as np
import pytest
from math import sqrt


# ---------------------------------------------------------------------------
# Test mesh fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_mesh():
    """Unit cube: 8 vertices, 12 triangles."""
    vertices = np.array([
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1],
    ], dtype=np.float32)
    triangles = np.array([
        [0,1,2], [0,2,3], [4,6,5], [4,7,6],
        [0,4,5], [0,5,1], [2,6,7], [2,7,3],
        [0,3,7], [0,7,4], [1,5,6], [1,6,2],
    ], dtype=np.int32)
    return vertices, triangles


@pytest.fixture
def icosphere_mesh():
    """Subdivided icosahedron: 642 vertices, 1280 triangles."""
    phi = (1 + sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float32)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    for _ in range(3):
        verts, tris = _subdivide(verts, tris)
    return verts, tris


def _subdivide(v, t):
    edge_midpoints = {}
    new_verts = list(v)
    new_tris = []
    for tri in t:
        mids = []
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            if e not in edge_midpoints:
                mid = (v[e[0]] + v[e[1]]) / 2
                mid /= np.linalg.norm(mid)
                edge_midpoints[e] = len(new_verts)
                new_verts.append(mid)
            mids.append(edge_midpoints[e])
        a, b, c = tri
        m0, m1, m2 = mids
        new_tris.extend([[a,m0,m2],[b,m1,m0],[c,m2,m1],[m0,m1,m2]])
    return np.array(new_verts, dtype=np.float32), np.array(new_tris, dtype=np.int32)


# ---------------------------------------------------------------------------
# Brute-force CPU reference
# ---------------------------------------------------------------------------

def _brute_force_unsigned_distance(point, vertices, triangles):
    """O(T) reference: compute distance to every triangle."""
    min_dsq = float('inf')
    min_idx = -1
    for i, tri in enumerate(triangles):
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        dsq = _point_triangle_distance_sq(point, a, b, c)
        if dsq < min_dsq:
            min_dsq = dsq
            min_idx = i
    return np.sqrt(min_dsq), min_idx


def _point_triangle_distance_sq(p, a, b, c):
    v21 = b - a; p1 = p - a
    v32 = c - b; p2 = p - b
    v13 = a - c; p3 = p - c
    nor = np.cross(v21, v13)
    s1 = np.sign(np.dot(np.cross(v21, nor), p1))
    s2 = np.sign(np.dot(np.cross(v32, nor), p2))
    s3 = np.sign(np.dot(np.cross(v13, nor), p3))
    if s1 + s2 + s3 < 2.0:
        e1 = v21 * np.clip(np.dot(v21, p1) / np.dot(v21, v21), 0, 1) - p1
        e2 = v32 * np.clip(np.dot(v32, p2) / np.dot(v32, v32), 0, 1) - p2
        e3 = v13 * np.clip(np.dot(v13, p3) / np.dot(v13, v13), 0, 1) - p3
        return min(np.dot(e1, e1), np.dot(e2, e2), np.dot(e3, e3))
    else:
        d = np.dot(nor, p1)
        return d * d / np.dot(nor, nor)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_build_from_numpy(self, cube_mesh):
        from mtlbvh import MtlBVH
        v, t = cube_mesh
        bvh = MtlBVH(v, t)
        assert bvh is not None

    def test_build_from_torch(self, cube_mesh):
        from mtlbvh import MtlBVH
        v, t = cube_mesh
        bvh = MtlBVH(torch.from_numpy(v), torch.from_numpy(t))
        assert bvh is not None

    def test_build_minimum_triangles(self):
        from mtlbvh import MtlBVH
        v = np.random.randn(12, 3).astype(np.float32)
        t = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],
                       [0,3,6],[1,4,7],[2,5,8],[0,1,3],
                       [3,6,9]], dtype=np.int32)
        bvh = MtlBVH(v, t)
        assert bvh is not None

    def test_build_too_few_triangles(self):
        from mtlbvh import MtlBVH
        v = np.random.randn(6, 3).astype(np.float32)
        t = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
        with pytest.raises(Exception):
            MtlBVH(v, t)

    def test_compat_alias(self, cube_mesh):
        from mtlbvh import cuBVH
        v, t = cube_mesh
        bvh = cuBVH(v, t)
        assert bvh is not None


# ---------------------------------------------------------------------------
# Unsigned distance
# ---------------------------------------------------------------------------

class TestUnsignedDistance:
    def test_origin_in_cube(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        d, fid, _ = bvh.unsigned_distance(p)
        assert abs(d.item() - 1.0) < 0.01

    def test_on_surface(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        d, fid, _ = bvh.unsigned_distance(p)
        assert d.item() < 0.01

    def test_outside(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.tensor([[0, 0, 3]], dtype=torch.float32)
        d, fid, _ = bvh.unsigned_distance(p)
        assert abs(d.item() - 2.0) < 0.01

    def test_parity_with_brute_force(self, icosphere_mesh):
        from mtlbvh import MtlBVH
        v, t = icosphere_mesh
        bvh = MtlBVH(v, t)
        torch.manual_seed(42)
        positions = torch.randn(500, 3) * 2
        d_bvh, _, _ = bvh.unsigned_distance(positions)
        for i in range(len(positions)):
            d_ref, _ = _brute_force_unsigned_distance(positions[i].numpy(), v, t)
            assert abs(d_bvh[i].item() - d_ref) < 0.01, \
                f"point {i}: bvh={d_bvh[i].item():.4f} ref={d_ref:.4f}"

    def test_batch_consistency(self, cube_mesh):
        """Batch results must match individual queries."""
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        torch.manual_seed(0)
        positions = torch.randn(50, 3) * 2
        d_batch, _, _ = bvh.unsigned_distance(positions)
        for i in range(len(positions)):
            d_single, _, _ = bvh.unsigned_distance(positions[i:i+1])
            assert abs(d_batch[i].item() - d_single.item()) < 1e-5, \
                f"point {i}: batch={d_batch[i].item():.6f} single={d_single.item():.6f}"

    def test_return_uvw(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.randn(10, 3)
        d, fid, uvw = bvh.unsigned_distance(p, return_uvw=True)
        assert uvw is not None
        assert uvw.shape == (10, 3)
        # Barycentric coords should sum to ~1
        assert (uvw.sum(dim=-1) - 1.0).abs().max() < 0.01

    def test_no_uvw_returns_none(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.randn(10, 3)
        d, fid, uvw = bvh.unsigned_distance(p, return_uvw=False)
        assert uvw is None


# ---------------------------------------------------------------------------
# Signed distance
# ---------------------------------------------------------------------------

class TestSignedDistance:
    def test_inside_negative(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        d, _, _ = bvh.signed_distance(p, mode='raystab')
        assert d.item() < 0

    def test_outside_positive(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.tensor([[0, 0, 3]], dtype=torch.float32)
        d, _, _ = bvh.signed_distance(p, mode='raystab')
        assert d.item() > 0

    def test_watertight_mode(self, icosphere_mesh):
        from mtlbvh import MtlBVH
        v, t = icosphere_mesh
        bvh = MtlBVH(v, t)
        # Points clearly inside/outside unit sphere
        inside = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        outside = torch.tensor([[0, 0, 3]], dtype=torch.float32)
        d_in, _, _ = bvh.signed_distance(inside, mode='watertight')
        d_out, _, _ = bvh.signed_distance(outside, mode='watertight')
        assert d_in.item() < 0
        assert d_out.item() > 0

    def test_raystab_mode(self, icosphere_mesh):
        from mtlbvh import MtlBVH
        v, t = icosphere_mesh
        bvh = MtlBVH(v, t)
        inside = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        outside = torch.tensor([[0, 0, 3]], dtype=torch.float32)
        d_in, _, _ = bvh.signed_distance(inside, mode='raystab')
        d_out, _, _ = bvh.signed_distance(outside, mode='raystab')
        assert d_in.item() < 0
        assert d_out.item() > 0

    def test_sign_agreement_watertight_vs_raystab(self, icosphere_mesh):
        from mtlbvh import MtlBVH
        v, t = icosphere_mesh
        bvh = MtlBVH(v, t)
        torch.manual_seed(123)
        positions = torch.randn(200, 3) * 1.5
        d_wt, _, _ = bvh.signed_distance(positions, mode='watertight')
        d_rs, _, _ = bvh.signed_distance(positions, mode='raystab')
        agreement = ((d_wt > 0) == (d_rs > 0)).float().mean()
        assert agreement > 0.95


# ---------------------------------------------------------------------------
# Ray tracing
# ---------------------------------------------------------------------------

class TestRayTrace:
    def test_hit_cube(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        ro = torch.tensor([[0, 0, -5]], dtype=torch.float32)
        rd = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        pos, fid, depth = bvh.ray_trace(ro, rd)
        assert abs(depth.item() - 4.0) < 0.01
        assert abs(pos[0, 2].item() - (-1.0)) < 0.01
        assert fid.item() >= 0

    def test_miss(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        ro = torch.tensor([[0, 0, -5]], dtype=torch.float32)
        rd = torch.tensor([[0, 1, 0]], dtype=torch.float32)  # parallel, misses
        pos, fid, depth = bvh.ray_trace(ro, rd)
        assert fid.item() == -1
        assert depth.item() > 100

    def test_sphere_hits(self, icosphere_mesh):
        from mtlbvh import MtlBVH
        v, t = icosphere_mesh
        bvh = MtlBVH(v, t)
        # Rays from outside pointing toward origin should all hit
        torch.manual_seed(7)
        origins = torch.randn(100, 3) * 3
        dirs = -origins / origins.norm(dim=-1, keepdim=True)
        pos, fid, depth = bvh.ray_trace(origins, dirs)
        hit_rate = (fid >= 0).float().mean()
        assert hit_rate > 0.95

    def test_batch_consistency(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        torch.manual_seed(0)
        ro = torch.randn(20, 3) * 3
        rd = -ro / ro.norm(dim=-1, keepdim=True)
        _, _, depth_batch = bvh.ray_trace(ro, rd)
        for i in range(len(ro)):
            _, _, depth_single = bvh.ray_trace(ro[i:i+1], rd[i:i+1])
            assert abs(depth_batch[i].item() - depth_single.item()) < 1e-4


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

class TestShapes:
    def test_flat_input(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.randn(100, 3)
        d, fid, _ = bvh.unsigned_distance(p)
        assert d.shape == (100,)
        assert fid.shape == (100,)

    def test_single_point(self, cube_mesh):
        from mtlbvh import MtlBVH
        bvh = MtlBVH(*cube_mesh)
        p = torch.randn(1, 3)
        d, fid, _ = bvh.unsigned_distance(p)
        assert d.shape == (1,)


# ---------------------------------------------------------------------------
# Degenerate triangles
# ---------------------------------------------------------------------------

class TestDegenerateTriangles:
    """Degenerate (zero-area) triangles must not produce NaN/Inf in queries."""

    @pytest.fixture
    def degenerate_mesh(self):
        """Mesh with a mix of valid and degenerate triangles.
        Needs >= 9 triangles for BVH construction.
        """
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],  # valid triangle 0
            [2, 0, 0], [3, 0, 0], [2, 1, 0],  # valid triangle 1
            [0, 0, 1], [1, 0, 1], [0, 1, 1],  # valid triangle 2
            [2, 0, 1], [3, 0, 1], [2, 1, 1],  # valid triangle 3
            [0, 0, 2], [1, 0, 2], [0, 1, 2],  # valid triangle 4
            [2, 0, 2], [3, 0, 2], [2, 1, 2],  # valid triangle 5
            [0, 0, 3], [1, 0, 3], [0, 1, 3],  # valid triangle 6
            [5, 5, 5], [5, 5, 5], [5, 5, 5],  # degenerate: all same point
            [6, 0, 0], [7, 0, 0], [6.5, 0, 0],  # degenerate: collinear
        ], dtype=np.float32)
        triangles = np.array([
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [9, 10, 11], [12, 13, 14], [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],  # degenerate: zero-area (same point)
            [24, 25, 26],  # degenerate: collinear
        ], dtype=np.int32)
        return vertices, triangles

    def test_unsigned_distance_no_nan(self, degenerate_mesh):
        from mtlbvh import MtlBVH
        v, t = degenerate_mesh
        bvh = MtlBVH(v, t)
        points = torch.tensor([
            [5, 5, 5],      # at degenerate point
            [6.5, 0, 0],    # on collinear edge
            [0.5, 0.5, 0],  # near valid triangle
            [10, 10, 10],   # far away
        ], dtype=torch.float32)
        d, fid, _ = bvh.unsigned_distance(points)
        assert not torch.isnan(d).any(), f"NaN in distances: {d}"
        assert not torch.isinf(d).any(), f"Inf in distances: {d}"
        assert (d >= 0).all(), f"Negative unsigned distances: {d}"

    def test_uvw_no_nan(self, degenerate_mesh):
        from mtlbvh import MtlBVH
        v, t = degenerate_mesh
        bvh = MtlBVH(v, t)
        points = torch.tensor([
            [5, 5, 5],
            [6.5, 0, 0],
            [0.5, 0.5, 0],
        ], dtype=torch.float32)
        d, fid, uvw = bvh.unsigned_distance(points, return_uvw=True)
        assert uvw is not None
        assert not torch.isnan(uvw).any(), f"NaN in barycentrics: {uvw}"
        assert not torch.isinf(uvw).any(), f"Inf in barycentrics: {uvw}"
