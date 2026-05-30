"""Tests for Metal serialization kernels (z-order and Hilbert)."""
import pytest
import torch
from flex_gemm import ops


@pytest.mark.parametrize("res", [4, 16, 64])
@pytest.mark.parametrize("mode", ["z_order", "hilbert"])
def test_encode_decode_roundtrip(device, sphere_coords_fn, res, mode):
    """Encode → decode roundtrip should recover original coords."""
    _, coords, shape = sphere_coords_fn(res)
    if coords.shape[0] == 0:
        pytest.skip("No coords at this resolution")

    codes = ops.serialize.encode_seq(coords, shape, mode=mode)
    decoded = ops.serialize.decode_seq(codes, shape, mode=mode)

    assert torch.all(coords == decoded), f"Roundtrip failed for res={res}, mode={mode}"


@pytest.mark.parametrize("mode", ["z_order", "hilbert"])
def test_encode_deterministic(device, sphere_coords_fn, mode):
    """Same input should produce same codes."""
    _, coords, shape = sphere_coords_fn(16)

    codes1 = ops.serialize.encode_seq(coords, shape, mode=mode)
    codes2 = ops.serialize.encode_seq(coords, shape, mode=mode)

    assert torch.all(codes1 == codes2), f"Non-deterministic encoding for mode={mode}"


@pytest.mark.parametrize("mode", ["z_order", "hilbert"])
def test_encode_unique(device, sphere_coords_fn, mode):
    """Each unique coord should produce a unique code."""
    _, coords, shape = sphere_coords_fn(16)

    codes = ops.serialize.encode_seq(coords, shape, mode=mode)
    assert codes.unique().shape[0] == codes.shape[0], f"Duplicate codes for mode={mode}"
