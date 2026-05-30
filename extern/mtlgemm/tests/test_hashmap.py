"""Tests for Metal hashmap kernels."""
import pytest
import torch
from flex_gemm import kernels


@pytest.mark.parametrize("res", [4, 16, 64])
def test_hashmap_insert_lookup_3d(device, sphere_coords_fn, res):
    """Insert 3D coords into hashmap, look them up, verify values match."""
    _, coords, shape = sphere_coords_fn(res)
    M = coords.shape[0]
    if M == 0:
        pytest.skip("No coords at this resolution")

    # uint32 keys, uint32 values
    hashmap_keys = torch.full((2 * M,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    hashmap_vals = torch.empty((2 * M,), dtype=torch.uint32, device=device)
    values = torch.randint(0, 2**31, (M,), device=device, dtype=torch.int32).to(torch.uint32)

    N, C, W, H, D = shape
    kernels.cuda.hashmap_insert_3d_cuda(hashmap_keys, hashmap_vals, coords, values, W, H, D)
    result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

    assert torch.all(values == result), f"Hashmap lookup mismatch at res={res}"


@pytest.mark.parametrize("res", [4, 16, 64])
def test_hashmap_insert_lookup_3d_u64_keys(device, sphere_coords_fn, res):
    """Same test with uint64 keys (for large grids)."""
    _, coords, shape = sphere_coords_fn(res)
    M = coords.shape[0]
    if M == 0:
        pytest.skip("No coords at this resolution")

    hashmap_keys = torch.full((2 * M,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    hashmap_vals = torch.empty((2 * M,), dtype=torch.uint32, device=device)
    values = torch.randint(0, 2**31, (M,), device=device, dtype=torch.int32).to(torch.uint32)

    N, C, W, H, D = shape
    kernels.cuda.hashmap_insert_3d_cuda(hashmap_keys, hashmap_vals, coords, values, W, H, D)
    result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

    assert torch.all(values == result), f"Hashmap u64 lookup mismatch at res={res}"


def test_hashmap_insert_3d_idx_as_val(device, sphere_coords_fn):
    """Insert with thread_id as value, verify sequential indices returned."""
    _, coords, shape = sphere_coords_fn(16)
    M = coords.shape[0]

    hashmap_keys = torch.full((2 * M,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    hashmap_vals = torch.empty((2 * M,), dtype=torch.uint32, device=device)

    N, C, W, H, D = shape
    kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)
    result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

    # Each coord should map to its index
    expected = torch.arange(M, dtype=torch.int64, device=device).to(torch.uint32)
    assert torch.all(result == expected), "idx_as_val mismatch"


def test_hashmap_out_of_bounds(device, sphere_coords_fn):
    """Lookup out-of-bounds coords should return 0xFFFFFFFF."""
    _, coords, shape = sphere_coords_fn(8)
    M = coords.shape[0]

    hashmap_keys = torch.full((2 * M,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    hashmap_vals = torch.empty((2 * M,), dtype=torch.uint32, device=device)

    N, C, W, H, D = shape
    kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

    # Create out-of-bounds coords
    oob_coords = torch.tensor([[0, -1, 0, 0], [0, W, 0, 0], [0, 0, H, 0]], dtype=torch.int32, device=device)
    result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, oob_coords, W, H, D)

    assert torch.all(result == torch.iinfo(torch.uint32).max), "OOB should return max uint32"
