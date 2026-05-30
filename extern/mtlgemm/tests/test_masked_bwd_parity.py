"""Numerical parity: MASKED_IMPLICIT_GEMM backward vs IMPLICIT_GEMM backward on MPS.

The masked bwd kernels compute grad_input and grad_weight with the same math
as the dense bwd — only the iteration order differs:
  - grad_input uses valid_kernel / valid_kernel_seg (same V-set as fwd under
    the change of variable u = V-1-v)
  - grad_weight uses valid_signal_i / valid_signal_o / valid_signal_seg (flat
    list of (n_input, n_output) pairs per kernel offset)

As with the forward parity test, fp32 stays bit-close (associative reductions
+ skip-zeroes equivalence), and fp16/bf16 stay within per-element epsilon of
a different summation order.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import pytest
import torch

from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="masked bwd parity is checked on MPS (production target).",
)


def _sphere_coords(res, ch, dtype, device):
    coords = torch.stack(torch.meshgrid(
        torch.arange(res), torch.arange(res), torch.arange(res), indexing="ij",
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.contiguous().to(device)
    feats = torch.randn(coords.shape[0], ch, dtype=dtype).to(device).contiguous()
    return feats, coords, torch.Size([1, ch, res, res, res])


# Per-dtype tolerances — see test_masked_vs_dense_parity.py for the rationale.
# fp32 atol scales with K (=V*Ci or V*Co, up to ~7000 for res=64 ch=256);
# masked reorders the summation so absolute error grows linearly in K while
# relative error stays at single-ULP float (~1e-7).
TOLERANCES = {
    torch.float32:  dict(atol=1e-3, rtol=1e-5),
    torch.float16:  dict(atol=3e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=8e-2, rtol=2e-2),
}


def _compare(name, a_mps, b_mps, tol, ctx):
    a_cpu = a_mps.detach().cpu().float()
    b_cpu = b_mps.detach().cpu().float()
    diff = (a_cpu - b_cpu).abs()
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    ref_max  = b_cpu.abs().max().item()
    rel_max  = max_err / max(ref_max, 1e-9)
    ok = torch.allclose(a_cpu, b_cpu, **tol)
    assert ok, (
        f"{name} parity failed ({ctx}): "
        f"max_err={max_err:.4e} rel_max={rel_max:.4e} mean_err={mean_err:.4e} "
        f"(allowed atol={tol['atol']}, rtol={tol['rtol']})"
    )


def _run_conv_with_algo(algo, feats, coords, shape, weight, bias, grad_out):
    """Call fwd then bwd directly to bypass autograd (local PyTorch build has a
    CUDA+MPS dispatch issue that blocks loss.backward() on MPS tensors — see
    FOLLOWUPS.md. This exercises the same kernels through the ops-level bwd
    method, which is what trellis2 calls under torch.autograd in production)."""
    from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction

    set_algorithm(algo)
    f = feats.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)

    # Forward: compute neighbor cache + output.
    Co, Kw, Kh, Kd, Ci = w.shape
    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), (1, 1, 1))
    out = SubMConv3dFunction._sparse_submanifold_conv_forward(f, cache, w, b)

    # Backward: directly call the bwd staticmethod with grad_out.
    grad_input, grad_weight, grad_bias = SubMConv3dFunction._sparse_submanifold_conv_backward(
        grad_out.contiguous(), f, cache, w, b,
    )
    return out, grad_input, grad_weight, grad_bias


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("res,ch", [
    (16, 32), (16, 64), (32, 64), (64, 64),
    (32, 128), (64, 128), (64, 256),
])
def test_masked_bwd_matches_dense(res, ch, dtype):
    torch.manual_seed(0xBADCAFE ^ res ^ (ch << 4))
    device = "mps"

    feats, coords, shape = _sphere_coords(res, ch, dtype, device)
    Co = ch
    Ks = 3
    weight = torch.randn(Co, Ks, Ks, Ks, ch, dtype=dtype).to(device)
    bias = torch.randn(Co, dtype=dtype).to(device)
    # Randomize grad_out to avoid accidental cancellations.
    grad_out = torch.randn(feats.shape[0], Co, dtype=dtype).to(device)

    dense_out, dense_gi, dense_gw, dense_gb = _run_conv_with_algo(
        Algorithm.IMPLICIT_GEMM, feats, coords, shape, weight, bias, grad_out,
    )
    masked_out, masked_gi, masked_gw, masked_gb = _run_conv_with_algo(
        Algorithm.MASKED_IMPLICIT_GEMM, feats, coords, shape, weight, bias, grad_out,
    )

    assert masked_gi.device.type == "mps"
    assert masked_gw.device.type == "mps"
    assert masked_gb.device.type == "mps"

    tol = TOLERANCES[dtype]
    ctx = f"res={res} ch={ch} dtype={dtype}"
    _compare("fwd output", masked_out, dense_out, tol, ctx)
    _compare("grad_input",  masked_gi, dense_gi, tol, ctx)
    _compare("grad_weight", masked_gw, dense_gw, tol, ctx)
    _compare("grad_bias",   masked_gb, dense_gb, tol, ctx)
