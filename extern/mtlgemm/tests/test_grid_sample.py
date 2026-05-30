"""Tests for Metal grid sample kernels."""
import pytest
import torch
from flex_gemm.ops.grid_sample import grid_sample_3d, grid_sample_3d_torch


@pytest.mark.parametrize("mode", ["nearest", "trilinear"])
def test_grid_sample_basic(device, sphere_coords_fn, mode):
    """Basic grid sample: query at known coord locations should return features."""
    feats, coords, shape = sphere_coords_fn(16, ch=8)
    if feats is None or coords.shape[0] == 0:
        pytest.skip("No coords")

    N = coords.shape[0]
    # Query at exact coord locations (should return exact features for nearest)
    query = coords[:min(N, 100), 1:].float().unsqueeze(0)  # [1, L, 3]

    out = grid_sample_3d(feats, coords, shape, query, mode=mode)

    assert out.shape == (1, query.shape[1], 8), f"Wrong output shape: {out.shape}"
    if mode == "nearest":
        # At exact coords, nearest should return exact features
        expected = feats[:min(N, 100)].unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-5), "Nearest lookup at exact coords should match"


def test_grid_sample_out_of_bounds(device, sphere_coords_fn):
    """Query points outside grid should return zero."""
    feats, coords, shape = sphere_coords_fn(8, ch=4)
    if feats is None:
        pytest.skip("No coords")

    oob_query = torch.tensor([[[-1.0, -1.0, -1.0], [100.0, 100.0, 100.0]]], device=device)
    out = grid_sample_3d(feats, coords, shape, oob_query, mode="nearest")
    assert torch.all(out == 0), "OOB query should return zero"


def test_grid_sample_trilinear_all_oob(device, sphere_coords_fn):
    """Trilinear with ALL 8 neighbors OOB should return zeros, not NaN/Inf."""
    feats, coords, shape = sphere_coords_fn(8, ch=4)
    if feats is None:
        pytest.skip("No coords")

    oob_query = torch.tensor([
        [[-1.0, -1.0, -1.0],
         [100.0, 100.0, 100.0],
         [-50.0, -50.0, -50.0],
         [999.0, 999.0, 999.0]],
    ], device=device)
    out = grid_sample_3d(feats, coords, shape, oob_query, mode="trilinear")
    assert not torch.isnan(out).any(), f"NaN in trilinear OOB output: {out}"
    assert not torch.isinf(out).any(), f"Inf in trilinear OOB output: {out}"
    assert torch.all(out == 0), f"OOB trilinear should return zero, got: {out}"


@pytest.mark.parametrize("mode", ["nearest", "trilinear"])
def test_grid_sample_u64_keys(device, mode):
    """Grid sample with uint64 hashmap keys (large spatial volume > 2^32)."""
    # shape with N*W*H*D > 2^32 triggers uint64 keys in init_hashmap
    W, H, D, ch = 2048, 2048, 2048, 4
    shape = torch.Size([1, ch, W, H, D])

    # Sparse coords in a small region of the large grid
    coords = torch.tensor([
        [0, 100, 100, 100],
        [0, 101, 100, 100],
        [0, 100, 101, 100],
        [0, 100, 100, 101],
        [0, 200, 200, 200],
    ], dtype=torch.int32, device=device)
    N = coords.shape[0]
    feats = torch.randn(N, ch, device=device)

    # Query at exact coord locations
    query = coords[:, 1:].float().unsqueeze(0)  # [1, N, 3]

    out = grid_sample_3d(feats, coords, shape, query, mode=mode)
    ref = grid_sample_3d_torch(feats, coords, shape, query, mode=mode)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    assert torch.allclose(out, ref, atol=1e-5), (
        f"u64 {mode} mismatch: max_err={( out - ref).abs().max().item():.6f}"
    )
