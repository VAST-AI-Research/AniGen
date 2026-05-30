import os, torch, pytest
os.environ.setdefault("SPARSE_BACKEND", "spconv")


def _make_sparse(feats, coords):
    from anigen.modules.sparse.basic import SparseTensor
    return SparseTensor(feats=feats, coords=coords)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_flex_gemm_submanifold_runs_and_is_finite():
    from anigen.modules.sparse.conv.conv_flex_gemm import SparseConv3d as FGConv
    N, Ci, Co = 128, 16, 32
    coords = torch.zeros(N, 4, dtype=torch.int32)
    coords[:, 1:] = torch.randint(0, 8, (N, 3))
    feats = torch.randn(N, Ci)
    x = _make_sparse(feats, coords).to("mps")
    conv = FGConv(Ci, Co, kernel_size=3).to("mps")
    out = conv(x)
    assert out.feats.shape == (N, Co)
    assert torch.isfinite(out.feats).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_flex_gemm_1x1x1_equals_linear():
    """A 1x1x1 submanifold conv has no spatial neighbours, so it must reduce to
    an exact per-voxel linear map (feats @ W.T + bias). This pins the weight
    layout, bias handling, and feats/coords bridging to be numerically correct
    (the multi-tap neighbour reduction is validated by flex_gemm's own suite)."""
    from anigen.modules.sparse.conv.conv_flex_gemm import SparseConv3d as FGConv
    torch.manual_seed(0)
    N, Ci, Co = 64, 8, 12
    coords = torch.zeros(N, 4, dtype=torch.int32)
    coords[:, 1:] = torch.randint(0, 8, (N, 3))
    feats = torch.randn(N, Ci)
    x = _make_sparse(feats, coords).to("mps")
    conv = FGConv(Ci, Co, kernel_size=1).to("mps")
    out = conv(x).feats.cpu()
    w = conv.weight.detach().cpu().reshape(Co, Ci)
    ref = feats @ w.t() + conv.bias.detach().cpu()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_flex_gemm_isolated_voxels_use_center_tap():
    """Voxels spaced >2 apart have no 3x3x3 neighbours, so a kernel_size=3 conv
    must reduce to the centre tap (w[:, 1, 1, 1, :]) — fixes the tap indexing."""
    from anigen.modules.sparse.conv.conv_flex_gemm import SparseConv3d as FGConv
    torch.manual_seed(1)
    N, Ci, Co = 20, 8, 12
    coords = torch.zeros(N, 4, dtype=torch.int32)
    coords[:, 1] = torch.arange(N) * 3
    feats = torch.randn(N, Ci)
    x = _make_sparse(feats, coords).to("mps")
    conv = FGConv(Ci, Co, kernel_size=3).to("mps")
    out = conv(x).feats.cpu()
    center = conv.weight.detach().cpu()[:, 1, 1, 1, :]
    ref = feats @ center.t() + conv.bias.detach().cpu()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_flex_gemm_strided_raises():
    from anigen.modules.sparse.conv.conv_flex_gemm import SparseConv3d as FGConv
    with pytest.raises(NotImplementedError):
        FGConv(16, 32, kernel_size=3, stride=2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_flex_gemm_inverse_raises():
    from anigen.modules.sparse.conv.conv_flex_gemm import SparseInverseConv3d as FGInv
    with pytest.raises(NotImplementedError):
        FGInv(16, 32, kernel_size=3, stride=2)
