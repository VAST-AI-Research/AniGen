"""Tests for sparse submanifold convolution forward pass."""
import pytest
import torch
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm


ALGO_ALL = [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM]


@pytest.mark.parametrize("algo", ALGO_ALL)
@pytest.mark.parametrize("kernel_size", [(3, 3, 3)])
def test_spconv_fwd(device, sphere_coords_fn, algo, kernel_size):
    """Forward pass: compare against manual im2col+matmul reference."""
    set_algorithm(algo)

    feats, coords, shape = sphere_coords_fn(16, ch=8)
    if feats is None or coords.shape[0] == 0:
        pytest.skip("No coords")

    N = coords.shape[0]
    Ci = 8
    Co = 16
    V = 27  # 3x3x3
    Kw, Kh, Kd = kernel_size
    weight = torch.randn(Co, Kw, Kh, Kd, Ci, device=device)
    bias = torch.randn(Co, device=device)

    output, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

    assert output.shape == (N, Co), f"Wrong output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    # Compare against EXPLICIT_GEMM reference
    set_algorithm(Algorithm.EXPLICIT_GEMM)
    ref_out, ref_cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
    nmap = ref_cache['neighbor_map']

    # Manual im2col reference
    im2col = torch.zeros((N * V, Ci), device=device, dtype=feats.dtype)
    mask = nmap.view(-1) != 0xFFFFFFFF
    im2col[mask] = feats[nmap.view(-1).long()[mask]]
    im2col = im2col.view(N, V * Ci)
    w = weight.view(Co, V * Ci).t()
    manual_ref = torch.addmm(bias, im2col, w)

    assert torch.allclose(output, manual_ref, atol=1e-5), (
        f"Fwd mismatch [{algo}]: max_err={(output - manual_ref).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("algo", ALGO_ALL)
@pytest.mark.parametrize("kernel_size", [(3, 3, 3)])
def test_spconv_fwd_neighbor_cache_reuse(device, sphere_coords_fn, algo, kernel_size):
    """Verify neighbor cache can be reused across calls."""
    set_algorithm(algo)

    feats, coords, shape = sphere_coords_fn(8, ch=4)
    if feats is None or coords.shape[0] == 0:
        pytest.skip("No coords")

    Ci = 4
    Co = 8
    Kw, Kh, Kd = kernel_size
    weight = torch.randn(Co, Kw, Kh, Kd, Ci, device=device)
    bias = torch.randn(Co, device=device)

    output1, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
    output2, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias, neighbor_cache=cache)

    assert torch.allclose(output1, output2, atol=1e-5), "Cache reuse should give same result"


def test_spconv_fwd_u64_keys(device):
    """Spconv with uint64 hashmap keys (large spatial volume > 2^32)."""
    set_algorithm(Algorithm.EXPLICIT_GEMM)

    W, H, D = 2048, 2048, 2048
    Ci, Co = 4, 8
    Kw, Kh, Kd = 3, 3, 3

    # Sparse coords in a small region of the large grid
    coords = torch.tensor([
        [0, 500, 500, 500],
        [0, 501, 500, 500],
        [0, 500, 501, 500],
        [0, 500, 500, 501],
        [0, 501, 501, 500],
        [0, 501, 500, 501],
        [0, 500, 501, 501],
        [0, 501, 501, 501],
    ], dtype=torch.int32, device=device)
    N = coords.shape[0]
    shape = torch.Size([1, Ci, W, H, D])
    feats = torch.randn(N, Ci, device=device)
    weight = torch.randn(Co, Kw, Kh, Kd, Ci, device=device)
    bias = torch.randn(Co, device=device)

    output, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

    assert output.shape == (N, Co), f"Wrong output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in u64 spconv output"
    assert not torch.isinf(output).any(), "Inf in u64 spconv output"

    # Verify against manual im2col reference
    V = Kw * Kh * Kd
    nmap = cache['neighbor_map']
    im2col = torch.zeros((N * V, Ci), device=device, dtype=feats.dtype)
    mask = nmap.view(-1) != 0xFFFFFFFF
    im2col[mask] = feats[nmap.view(-1).long()[mask]]
    im2col = im2col.view(N, V * Ci)
    w = weight.view(Co, V * Ci).t()
    manual_ref = torch.addmm(bias, im2col, w)

    assert torch.allclose(output, manual_ref, atol=1e-5), (
        f"u64 spconv mismatch: max_err={(output - manual_ref).abs().max().item():.2e}"
    )
