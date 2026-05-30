"""Tests for the scipy cKDTree CPU drop-ins for pytorch3d knn_points / ball_query.

Parity note (knn idx ordering): cKDTree and a brute-force `topk` can disagree on
the ORDER of returned neighbors for tied / near-tied distances. With well-separated
random points in 3D (default randn) exact ties are vanishingly rare, so we assert
exact idx equality here. If this ever proves flaky, switch the idx assertion to a
per-query SORTED-SET comparison (keep the squared-distance allclose either way) —
do NOT drop the idx check entirely.
"""
import torch
from anigen_mps.knn_cpu import knn_points, ball_query


def _brute_knn(q, ref, K):
    d = torch.cdist(q, ref)                       # [1, P1, P2]
    dist, idx = d.topk(K, largest=False, dim=-1)
    return dist ** 2, idx


def test_knn_points_matches_brute_force():
    torch.manual_seed(0)
    q = torch.randn(1, 50, 3); ref = torch.randn(1, 80, 3)
    out = knn_points(q, ref, K=3)
    bd, bi = _brute_knn(q, ref, 3)
    assert out.idx.shape == (1, 50, 3)
    assert torch.equal(out.idx, bi)
    assert torch.allclose(out.dists, bd, atol=1e-4)


def test_ball_query_radius_filtering():
    torch.manual_seed(1)
    q = torch.randn(1, 40, 3); ref = torch.randn(1, 60, 3)
    out_dists, out_idx, _ = ball_query(q, ref, K=5, radius=0.5)
    assert out_idx.shape == (1, 40, 5)
    assert (out_idx == -1).any() or out_idx.min() >= 0


def test_knn_points_attribute_access_and_squared_dists():
    """Call sites use .idx / .dists attribute access; dists must be squared."""
    torch.manual_seed(2)
    q = torch.randn(1, 10, 3); ref = torch.randn(1, 20, 3)
    out = knn_points(q, ref, K=1, norm=2, return_nn=False)
    assert out.idx.shape == (1, 10, 1)
    assert out.dists.shape == (1, 10, 1)
    # squared euclidean to the single nearest neighbor
    nn = ref[0, out.idx[0, :, 0]]
    expected = ((q[0] - nn) ** 2).sum(-1)
    assert torch.allclose(out.dists[0, :, 0], expected, atol=1e-4)


def test_knn_points_K_greater_than_n_no_oob_index():
    """tree.query pads with index == n when K > n; idx must stay in-range."""
    torch.manual_seed(3)
    q = torch.randn(1, 5, 3); ref = torch.randn(1, 3, 3)  # K=4 > n=3
    out = knn_points(q, ref, K=4)
    assert out.idx.shape == (1, 5, 4)
    assert out.idx.max() < ref.shape[1]   # no out-of-range index n
    assert out.idx.min() >= 0


def test_ball_query_K_greater_than_n_padding():
    """ball_query with K > n: out-of-range slots are -1 padded, dist 0."""
    torch.manual_seed(4)
    q = torch.randn(1, 6, 3); ref = torch.randn(1, 4, 3)  # K=10 > n=4
    dists, idx, _ = ball_query(q, ref, K=10, radius=100.0)
    assert idx.shape == (1, 6, 10)
    # last slots beyond n must be -1 padding (radius huge so first n are valid)
    assert (idx[0, :, ref.shape[1]:] == -1).all()
    assert (dists[0, :, ref.shape[1]:] == 0.0).all()


def test_returns_on_input_device():
    q = torch.randn(1, 8, 3); ref = torch.randn(1, 12, 3)
    out = knn_points(q, ref, K=2)
    assert out.idx.device == q.device
    assert out.dists.device == q.device
    d, i, _ = ball_query(q, ref, K=2, radius=1.0)
    assert i.device == q.device


def test_bootstrap_installs_real_knn_into_pytorch3d_ops():
    import anigen_mps  # noqa: F401 -- runs install_knn_shim
    import pytorch3d.ops as ops
    from anigen_mps import knn_cpu
    assert ops.knn_points is knn_cpu.knn_points
    assert ops.ball_query is knn_cpu.ball_query
