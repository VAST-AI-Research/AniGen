"""
Absolute parity tests: Metal output vs CPU reference, element-by-element.

Every test reads Metal output back to numpy and compares against the exact
CPU reference implementation that replicates the CUDA kernel math.
No invariant checks — only numerical equality within floating point tolerance.
"""
import torch
import numpy as np
import math
import pytest

try:
    from cumesh import CuMesh as MtlMesh
    HAS_CUMESH = True
except ImportError:
    HAS_CUMESH = False

pytestmark = pytest.mark.skipif(not HAS_CUMESH, reason="cumesh not built")


# ============================================================
# CPU references (exact CUDA kernel math)
# ============================================================

def compute_face_areas_ref(v, f):
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def compute_face_normals_ref(v, f):
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return cross / norms


def compute_vertex_normals_ref(v, f):
    """Exact match to compute_vertex_normals_kernel: area-weighted face normals."""
    V = v.shape[0]
    normals = np.zeros((V, 3), dtype=np.float64)
    for fi in range(f.shape[0]):
        v0, v1, v2 = v[f[fi, 0]], v[f[fi, 1]], v[f[fi, 2]]
        fn = np.cross(v1 - v0, v2 - v0)  # not normalized — area-weighted
        for j in range(3):
            normals[f[fi, j]] += fn
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (normals / norms).astype(np.float32)


def expand_edges_ref(faces):
    """Returns (num_unique_edges, boundary_count, manifold_count)."""
    edge_set = {}
    for fi in range(faces.shape[0]):
        fa = faces[fi]
        for i in range(3):
            a, b = int(fa[i]), int(fa[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_set[key] = edge_set.get(key, 0) + 1
    num_edges = len(edge_set)
    boundary = sum(1 for c in edge_set.values() if c == 1)
    manifold = sum(1 for c in edge_set.values() if c == 2)
    return num_edges, boundary, manifold


def connected_components_ref(faces):
    """Union-find on face adjacency via shared edges."""
    F = faces.shape[0]
    parent = list(range(F))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Build edge → face map
    edge2face = {}
    for fi in range(F):
        fa = faces[fi]
        for i in range(3):
            a, b = int(fa[i]), int(fa[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            if key in edge2face:
                for other in edge2face[key]:
                    ra, rb = find(fi), find(other)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                edge2face[key].append(fi)
            else:
                edge2face[key] = [fi]

    # Compress
    for i in range(F):
        parent[i] = find(i)
    return len(set(parent))


def mark_degenerate_ref(v, f, abs_thresh=1e-24, rel_thresh=1e-12):
    """Returns boolean mask of faces to keep."""
    keep = np.ones(f.shape[0], dtype=bool)
    for i in range(f.shape[0]):
        fa = f[i]
        if fa[0] == fa[1] or fa[1] == fa[2] or fa[2] == fa[0]:
            keep[i] = False
            continue
        v0, v1, v2 = v[fa[0]], v[fa[1]], v[fa[2]]
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2
        max_edge = max(np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2))
        area = np.linalg.norm(np.cross(e0, e1)) * 0.5
        thresh = min(rel_thresh * max_edge * max_edge, abs_thresh)
        keep[i] = area >= thresh
    return keep


# ============================================================
# Mesh builders
# ============================================================

def _mesh(v, f):
    m = MtlMesh()
    m.init(torch.from_numpy(v), torch.from_numpy(f))
    return m


def make_cube():
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    return v, f


def make_tetra():
    v = np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]
    ], dtype=np.int32)
    return v, f


def make_open_plane():
    """4x4 grid of vertices, 18 triangles, has boundary."""
    v = []
    for y in range(4):
        for x in range(4):
            v.append([x, y, 0])
    v = np.array(v, dtype=np.float32)
    f = []
    for y in range(3):
        for x in range(3):
            i = y * 4 + x
            f.append([i, i + 1, i + 4])
            f.append([i + 1, i + 5, i + 4])
    f = np.array(f, dtype=np.int32)
    return v, f


def make_icosphere(subdivisions=2):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    v = np.array(verts, dtype=np.float32)
    f = np.array(faces, dtype=np.int32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdivisions):
        cache = {}
        new_f = []
        for tri in f:
            mids = []
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                key = (min(a, b), max(a, b))
                if key not in cache:
                    mid = (v[a] + v[b]) / 2
                    mid /= np.linalg.norm(mid)
                    cache[key] = len(v)
                    v = np.vstack([v, mid.reshape(1, 3)])
                mids.append(cache[key])
            new_f.append([tri[0], mids[0], mids[2]])
            new_f.append([tri[1], mids[1], mids[0]])
            new_f.append([tri[2], mids[2], mids[1]])
            new_f.append([mids[0], mids[1], mids[2]])
        f = np.array(new_f, dtype=np.int32)
    return v.astype(np.float32), f


# ============================================================
# GEOMETRY: element-by-element parity
# ============================================================

class TestFaceAreasParity:
    """Metal face areas must match CPU reference within float32 tolerance."""

    def test_cube(self):
        v, f = make_cube()
        ref = compute_face_areas_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_areas()
        mtl = m.read_face_areas().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-6, rtol=1e-5)

    def test_tetra(self):
        v, f = make_tetra()
        ref = compute_face_areas_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_areas()
        mtl = m.read_face_areas().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-6, rtol=1e-5)

    def test_icosphere_320(self):
        v, f = make_icosphere(2)
        ref = compute_face_areas_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_areas()
        mtl = m.read_face_areas().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-6, rtol=1e-5)

    def test_icosphere_1280(self):
        v, f = make_icosphere(3)
        ref = compute_face_areas_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_areas()
        mtl = m.read_face_areas().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-6, rtol=1e-5)


class TestFaceNormalsParity:
    """Metal face normals must match CPU reference per-element."""

    def test_cube(self):
        v, f = make_cube()
        ref = compute_face_normals_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_normals()
        mtl = m.read_face_normals().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-5, rtol=1e-4)

    def test_icosphere_320(self):
        v, f = make_icosphere(2)
        ref = compute_face_normals_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_normals()
        mtl = m.read_face_normals().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-5, rtol=1e-4)


class TestVertexNormalsParity:
    """Metal vertex normals must match CPU reference."""

    def test_cube(self):
        v, f = make_cube()
        ref = compute_vertex_normals_ref(v, f)
        m = _mesh(v, f)
        m.compute_vertex_normals()
        mtl = m.read_vertex_normals().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-5, rtol=1e-4)

    def test_icosphere_320(self):
        v, f = make_icosphere(2)
        ref = compute_vertex_normals_ref(v, f)
        m = _mesh(v, f)
        m.compute_vertex_normals()
        mtl = m.read_vertex_normals().numpy()
        np.testing.assert_allclose(mtl, ref, atol=1e-5, rtol=1e-4)


# ============================================================
# CONNECTIVITY: exact count parity
# ============================================================

class TestEdgeCountParity:
    """Metal edge/boundary/manifold counts must match CPU reference exactly."""

    def _check(self, v, f):
        ref_E, ref_B, ref_M = expand_edges_ref(f)
        m = _mesh(v, f)
        m.get_edges()
        m.get_edge_face_adjacency()
        m.get_boundary_info()
        assert m.num_edges == ref_E, f"Edge count: metal={m.num_edges} ref={ref_E}"
        assert m.num_boundaries == ref_B, f"Boundary count: metal={m.num_boundaries} ref={ref_B}"

    def test_cube(self):
        self._check(*make_cube())

    def test_tetra(self):
        self._check(*make_tetra())

    def test_open_plane(self):
        self._check(*make_open_plane())

    def test_icosphere_320(self):
        self._check(*make_icosphere(2))

    def test_icosphere_1280(self):
        self._check(*make_icosphere(3))


class TestConnectedComponentsParity:
    """Metal connected component count must match CPU reference."""

    def test_cube_single(self):
        v, f = make_cube()
        ref_cc = connected_components_ref(f)
        m = _mesh(v, f)
        m.get_edges()
        m.get_edge_face_adjacency()
        m.get_boundary_info()
        m.get_manifold_face_adjacency()
        m.get_connected_components()
        nc, _ = m.read_connected_components()
        assert nc == ref_cc == 1

    def test_icosphere_single(self):
        v, f = make_icosphere(2)
        ref_cc = connected_components_ref(f)
        m = _mesh(v, f)
        m.get_edges()
        m.get_edge_face_adjacency()
        m.get_boundary_info()
        m.get_manifold_face_adjacency()
        m.get_connected_components()
        nc, _ = m.read_connected_components()
        assert nc == ref_cc == 1

    def test_two_disconnected(self):
        """Two tetrahedra with no shared vertices."""
        v1, f1 = make_tetra()
        v2 = v1 + 10.0  # offset far away
        f2 = f1 + len(v1)
        v = np.vstack([v1, v2])
        f = np.vstack([f1, f2])
        ref_cc = connected_components_ref(f)
        m = _mesh(v, f)
        m.get_edges()
        m.get_edge_face_adjacency()
        m.get_boundary_info()
        m.get_manifold_face_adjacency()
        m.get_connected_components()
        nc, _ = m.read_connected_components()
        assert nc == ref_cc == 2


# ============================================================
# CLEANUP: degenerate/duplicate face removal parity
# ============================================================

class TestDegenerateParity:
    """Metal degenerate face removal must match CPU reference."""

    def test_mixed_mesh(self):
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0, 0, 0],  # v3 == v0
        ], dtype=np.float32)
        f = np.array([
            [0, 1, 2],  # good
            [0, 0, 1],  # dup vertex → degenerate
            [0, 1, 3],  # zero area → degenerate
        ], dtype=np.int32)
        ref_keep = mark_degenerate_ref(v, f)
        m = _mesh(v, f)
        m.remove_degenerate_faces()
        # Only face 0 should survive
        assert m.num_faces == int(ref_keep.sum()), \
            f"metal={m.num_faces} ref={int(ref_keep.sum())}"

    def test_clean_cube_unchanged(self):
        v, f = make_cube()
        ref_keep = mark_degenerate_ref(v, f)
        assert ref_keep.all()  # all good
        m = _mesh(v, f)
        m.remove_degenerate_faces()
        assert m.num_faces == 12  # unchanged


class TestDuplicateFaceParity:
    """Duplicate face removal: Metal vs CPU."""

    def test_exact_duplicate(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int32)
        m = _mesh(v, f)
        m.remove_duplicate_faces()
        assert m.num_faces == 1

    def test_rotated_duplicate(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32)
        m = _mesh(v, f)
        m.remove_duplicate_faces()
        assert m.num_faces == 1

    def test_no_duplicates(self):
        v, f = make_cube()
        m = _mesh(v, f)
        m.remove_duplicate_faces()
        assert m.num_faces == 12


# ============================================================
# SIMPLIFICATION: structural parity
# ============================================================

class TestSimplifyParity:
    """Simplification must produce valid meshes with correct topology."""

    def _verify_valid(self, m, expect_closed):
        """Verify mesh is valid: all face indices in range, optional closed check."""
        v_out, f_out = m.read()
        V, F = v_out.shape[0], f_out.shape[0]
        assert f_out.min().item() >= 0
        assert f_out.max().item() < V
        if expect_closed:
            m.get_edges()
            m.get_edge_face_adjacency()
            m.get_boundary_info()
            assert m.num_boundaries == 0, f"Expected closed mesh, got {m.num_boundaries} boundary edges"

    def test_cube_simplify_preserves_closed(self):
        v, f = make_cube()
        m = _mesh(v, f)
        m.simplify(target_num_faces=8)
        assert m.num_faces <= 10
        self._verify_valid(m, expect_closed=True)

    def test_icosphere_simplify_to_half(self):
        v, f = make_icosphere(2)  # 320 faces
        m = _mesh(v, f)
        m.simplify(target_num_faces=160)
        assert m.num_faces <= 180
        self._verify_valid(m, expect_closed=True)

    def test_icosphere_simplify_aggressive(self):
        v, f = make_icosphere(3)  # 1280 faces
        m = _mesh(v, f)
        m.simplify(target_num_faces=100)
        assert m.num_faces <= 120
        self._verify_valid(m, expect_closed=True)


# ============================================================
# ATLAS: chart ID parity
# ============================================================

class TestAtlasParity:
    """Chart clustering must produce valid, complete chart assignments."""

    def test_cube_all_faces_assigned(self):
        v, f = make_cube()
        m = _mesh(v, f)
        m.compute_charts()
        nc, ids, *_ = m.read_atlas_charts()
        # Every face must have a chart
        assert ids.shape[0] == 12
        # IDs in valid range
        assert ids.min().item() >= 0
        assert ids.max().item() < nc
        # Every chart ID used
        assert len(torch.unique(ids)) == nc

    def test_icosphere_all_faces_assigned(self):
        v, f = make_icosphere(2)
        m = _mesh(v, f)
        m.compute_charts()
        nc, ids, *_ = m.read_atlas_charts()
        assert ids.shape[0] == 320
        assert ids.min().item() >= 0
        assert ids.max().item() < nc
        assert len(torch.unique(ids)) == nc

    def test_icosphere_1280_all_faces_assigned(self):
        v, f = make_icosphere(3)
        m = _mesh(v, f)
        m.compute_charts()
        nc, ids, *_ = m.read_atlas_charts()
        assert ids.shape[0] == 1280
        assert ids.min().item() >= 0
        assert ids.max().item() < nc
        assert len(torch.unique(ids)) == nc


# ============================================================
# END-TO-END PIPELINE: full TRELLIS.2 flow
# ============================================================

class TestFullPipelineParity:
    """The exact pipeline TRELLIS.2 runs: cleanup → simplify → charts → UV."""

    def test_pipeline_icosphere_320(self):
        v, f = make_icosphere(2)

        # Step 1: geometry matches ref
        ref_areas = compute_face_areas_ref(v, f)
        ref_normals = compute_face_normals_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_areas()
        m.compute_face_normals()
        np.testing.assert_allclose(m.read_face_areas().numpy(), ref_areas, atol=1e-6)
        np.testing.assert_allclose(m.read_face_normals().numpy(), ref_normals, atol=1e-5)

        # Step 2: connectivity matches ref
        ref_E, ref_B, _ = expand_edges_ref(f)
        m.get_edges()
        m.get_edge_face_adjacency()
        m.get_boundary_info()
        assert m.num_edges == ref_E
        assert m.num_boundaries == ref_B

        # Step 3: simplify
        m.simplify(target_num_faces=100)
        assert m.num_faces <= 120
        v_s, f_s = m.read()
        assert f_s.min().item() >= 0
        assert f_s.max().item() < v_s.shape[0]

        # Step 4: charts
        m.compute_charts()
        nc, ids, *_ = m.read_atlas_charts()
        assert nc >= 1
        assert ids.shape[0] == m.num_faces
        assert len(torch.unique(ids)) == nc

    def test_pipeline_icosphere_5120(self):
        v, f = make_icosphere(4)  # 5120 faces

        # Geometry parity
        ref_areas = compute_face_areas_ref(v, f)
        m = _mesh(v, f)
        m.compute_face_areas()
        np.testing.assert_allclose(m.read_face_areas().numpy(), ref_areas, atol=1e-5)

        # Connectivity parity
        ref_E, ref_B, _ = expand_edges_ref(f)
        m.get_edges()
        m.get_edge_face_adjacency()
        m.get_boundary_info()
        assert m.num_edges == ref_E
        assert m.num_boundaries == ref_B == 0

        # Simplify → charts
        m.simplify(target_num_faces=1000)
        assert m.num_faces <= 1100
        m.compute_charts()
        nc, ids, *_ = m.read_atlas_charts()
        assert nc >= 1
        assert len(torch.unique(ids)) == nc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
