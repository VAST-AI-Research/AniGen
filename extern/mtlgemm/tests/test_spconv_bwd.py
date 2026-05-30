"""Tests for sparse submanifold convolution backward pass."""
import pytest
import torch
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm


ALGO_ALL = [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM]


@pytest.mark.parametrize("algo", [Algorithm.EXPLICIT_GEMM])
def test_spconv_bwd_gradcheck(device, sphere_coords_fn, algo):
    """Verify backward pass correctness with torch.autograd.gradcheck.
    Only EXPLICIT_GEMM: Metal kernels are float32-only, float64 produces inf."""
    set_algorithm(algo)

    feats, coords, shape = sphere_coords_fn(8, ch=4)
    if feats is None or coords.shape[0] < 2:
        pytest.skip("Not enough coords")

    # Use small subset for gradcheck (expensive)
    N = min(coords.shape[0], 50)
    feats = feats[:N].clone().detach().requires_grad_(True).double()
    coords = coords[:N].clone()

    Ci = 4
    Co = 4
    weight = torch.randn(Co, 3, 3, 3, Ci, device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(Co, device=device, dtype=torch.float64, requires_grad=True)

    def fn(f, w, b):
        out, _ = sparse_submanifold_conv3d(f, coords, shape, w, b)
        return out

    torch.autograd.gradcheck(fn, (feats, weight, bias), eps=1e-3, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("algo", ALGO_ALL)
def test_spconv_bwd_grad_values(device, sphere_coords_fn, algo):
    """Backward pass: compare gradient values against EXPLICIT_GEMM reference."""
    set_algorithm(algo)

    feats, coords, shape = sphere_coords_fn(16, ch=8)
    if feats is None or coords.shape[0] == 0:
        pytest.skip("No coords")

    weight = torch.randn(16, 3, 3, 3, 8, device=device)
    bias = torch.randn(16, device=device)

    # Run target algorithm
    feats_a = feats.clone().requires_grad_(True)
    weight_a = weight.clone().requires_grad_(True)
    bias_a = bias.clone().requires_grad_(True)
    set_algorithm(algo)
    output_a, _ = sparse_submanifold_conv3d(feats_a, coords, shape, weight_a, bias_a)
    grad_out = torch.randn_like(output_a)
    output_a.backward(grad_out)

    # Run EXPLICIT_GEMM reference
    feats_r = feats.clone().requires_grad_(True)
    weight_r = weight.clone().requires_grad_(True)
    bias_r = bias.clone().requires_grad_(True)
    set_algorithm(Algorithm.EXPLICIT_GEMM)
    output_r, _ = sparse_submanifold_conv3d(feats_r, coords, shape, weight_r, bias_r)
    output_r.backward(grad_out)

    assert feats_a.grad is not None, "No gradient for feats"
    assert weight_a.grad is not None, "No gradient for weight"
    assert bias_a.grad is not None, "No gradient for bias"
    assert feats_a.grad.shape == feats.shape
    assert weight_a.grad.shape == weight.shape

    assert torch.allclose(feats_a.grad, feats_r.grad, atol=1e-5), (
        f"grad_input mismatch [{algo}]: max_err={(feats_a.grad - feats_r.grad).abs().max().item():.2e}"
    )
    assert torch.allclose(weight_a.grad, weight_r.grad, atol=1e-5), (
        f"grad_weight mismatch [{algo}]: max_err={(weight_a.grad - weight_r.grad).abs().max().item():.2e}"
    )
    assert torch.allclose(bias_a.grad, bias_r.grad, atol=1e-5), (
        f"grad_bias mismatch [{algo}]"
    )
