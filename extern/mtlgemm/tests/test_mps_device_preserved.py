"""End-to-end smoke test for the Metal backend on MPS.

Originally added after an external user (shivampkumar) reported that running
SPARSE_CONV_BACKEND=flex_gemm on TRELLIS.2 crashed at the next LayerNorm with
"Passed CPU tensor to MPS op". The root cause was that every Metal-backed op
was returning a CPU tensor regardless of input device. These tests exist to
make that class of regression impossible to ship again.
"""

import os
import pytest
import torch

import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm

# Silence the masked-implicit-GEMM-not-implemented warning during tests.
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

mps_required = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available on this machine",
)


def _sphere_coords(res: int, ch: int, device: str, dtype: torch.dtype):
    """Build a sparse-shell coords tensor. Sphere construction runs on CPU
    (PyTorch MPS lacks kernels for some int ops on tensors of these dtypes
    in 2.10), then the result is moved to the target device."""
    coords = torch.stack(torch.meshgrid(
        torch.arange(res),
        torch.arange(res),
        torch.arange(res),
        indexing="ij",
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.contiguous().to(device)
    feats = torch.randn(coords.shape[0], ch, dtype=dtype).to(device).contiguous()
    return feats, coords, torch.Size([1, ch, res, res, res])


def _run_spconv(device: str, dtype: torch.dtype):
    flex_gemm.ops.spconv.set_algorithm(Algorithm.IMPLICIT_GEMM)
    Ci, Co, Ks = 32, 32, 3
    feats, coords, shape = _sphere_coords(res=16, ch=Ci, device=device, dtype=dtype)
    weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=dtype, device=device)
    bias = torch.randn(Co, dtype=dtype, device=device)
    out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
    return out


@mps_required
def test_mps_spconv_returns_mps_tensor_fp32():
    """The exact bug from the original issue: output came back on CPU."""
    out = _run_spconv("mps", torch.float32)
    assert out.device.type == "mps", f"Expected MPS output, got {out.device}"
    assert out.dtype == torch.float32


@mps_required
def test_mps_spconv_returns_mps_tensor_fp16():
    out = _run_spconv("mps", torch.float16)
    assert out.device.type == "mps"
    assert out.dtype == torch.float16


@mps_required
def test_mps_spconv_returns_mps_tensor_bf16():
    out = _run_spconv("mps", torch.bfloat16)
    assert out.device.type == "mps"
    assert out.dtype == torch.bfloat16


@mps_required
def test_mps_spconv_output_feeds_layernorm_without_crash():
    """The LayerNorm crash was the user-visible symptom. Regress against it
    using F.layer_norm directly to avoid PyTorch-build-version issues with
    nn.LayerNorm parameter initialization on some MPS builds."""
    out = _run_spconv("mps", torch.float16)
    # Build LN params on CPU then move to MPS — same shape as TRELLIS uses.
    weight = torch.ones(out.shape[-1], dtype=torch.float16).to("mps")
    bias = torch.zeros(out.shape[-1], dtype=torch.float16).to("mps")
    out2 = torch.nn.functional.layer_norm(out, (out.shape[-1],), weight, bias)
    assert out2.device.type == "mps"
    # Force a sync to surface any deferred MPS errors. This is exactly the
    # path that hit "Passed CPU tensor to MPS op" before the device-routing fix.
    _ = out2.sum().item()


@mps_required
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mps_matches_cpu(dtype):
    """Numerical parity check between CPU and MPS Metal kernels."""
    flex_gemm.ops.spconv.set_algorithm(Algorithm.IMPLICIT_GEMM)
    torch.manual_seed(0)
    Ci, Co, Ks = 32, 32, 3
    feats_cpu, coords_cpu, shape = _sphere_coords(res=12, ch=Ci, device="cpu", dtype=dtype)
    weight_cpu = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=dtype, device="cpu")
    bias_cpu = torch.randn(Co, dtype=dtype, device="cpu")

    feats_mps = feats_cpu.to("mps")
    coords_mps = coords_cpu.to("mps")
    weight_mps = weight_cpu.to("mps")
    bias_mps = bias_cpu.to("mps")

    out_cpu, _ = sparse_submanifold_conv3d(feats_cpu, coords_cpu, shape, weight_cpu, bias_cpu)
    out_mps, _ = sparse_submanifold_conv3d(feats_mps, coords_mps, shape, weight_mps, bias_mps)

    assert out_mps.device.type == "mps"
    diff = (out_cpu.float() - out_mps.cpu().float()).abs()
    # fp32 tolerance is tight; fp16/bf16 accumulators stay fp32 so error stays small.
    tol = 5e-3 if dtype == torch.float16 else 1e-4
    assert diff.max().item() < tol, (
        f"CPU vs MPS divergence for {dtype}: max |diff|={diff.max().item():.3e} "
        f"(threshold {tol:.0e})"
    )
