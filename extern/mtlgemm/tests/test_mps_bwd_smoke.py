"""MPS backward smoke tests — a tiny, fast gate that each bwd path produces
grads on the MPS device with finite values and matches a CPU run on the same
inputs within tolerance. Covers both IMPLICIT_GEMM and MASKED_IMPLICIT_GEMM
for fp32 / fp16 / bf16.

Rationale: before these tests, mtlgemm had zero MPS backward coverage in CI —
the only parity we ran was forward. A regression in the dispatch path or the
output-allocation (the two classes of bug that shipped in rounds 1 and 2)
would only surface at end-to-end run time.

This test does NOT invoke torch.autograd.backward() because the user's local
PyTorch build has a CUDA+MPS dispatch issue unrelated to this repo. It calls
the module's bwd staticmethod directly — same code path trellis2 hits under
autograd in production.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import pytest
import torch

from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS smoke tests require MPS.",
)


def _small_coords(res, ch, dtype, device):
    """Tiny sparse volume — the smoke test is about path coverage, not size."""
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


def _run(algo, feats, coords, shape, weight, bias, grad_out):
    set_algorithm(algo)
    f = feats.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    Co, Kw, Kh, Kd, Ci = w.shape
    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), (1, 1, 1))
    out = SubMConv3dFunction._sparse_submanifold_conv_forward(f, cache, w, b)
    gi, gw, gb = SubMConv3dFunction._sparse_submanifold_conv_backward(
        grad_out.contiguous(), f, cache, w, b,
    )
    return out, gi, gw, gb


SMOKE_TOL = {
    torch.float32:  dict(atol=5e-4, rtol=1e-4),
    torch.float16:  dict(atol=5e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=2e-2),
}


@needs_mps
@pytest.mark.parametrize("algo", [Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mps_bwd_smoke(algo, dtype):
    """For each algorithm × dtype: run bwd on MPS and on CPU, assert the MPS
    grads (a) are on MPS, (b) are finite, (c) match the CPU run within tol."""
    torch.manual_seed(0x5CAFE ^ hash(algo) ^ hash(dtype))
    res, ch = 16, 32

    feats_m, coords_m, shape = _small_coords(res, ch, dtype, "mps")
    weight_m = torch.randn(ch, 3, 3, 3, ch, dtype=dtype).to("mps")
    bias_m = torch.randn(ch, dtype=dtype).to("mps")
    grad_out_m = torch.randn(feats_m.shape[0], ch, dtype=dtype).to("mps")

    # MPS run — the one we actually care about.
    _, gi_m, gw_m, gb_m = _run(algo, feats_m, coords_m, shape, weight_m, bias_m, grad_out_m)

    assert gi_m.device.type == "mps", f"grad_input on {gi_m.device}"
    assert gw_m.device.type == "mps", f"grad_weight on {gw_m.device}"
    assert gb_m.device.type == "mps", f"grad_bias on {gb_m.device}"

    gi_cpu = gi_m.detach().cpu().float()
    gw_cpu = gw_m.detach().cpu().float()
    gb_cpu = gb_m.detach().cpu().float()
    assert torch.isfinite(gi_cpu).all(), "grad_input has non-finite values"
    assert torch.isfinite(gw_cpu).all(), "grad_weight has non-finite values"
    assert torch.isfinite(gb_cpu).all(), "grad_bias has non-finite values"

    # CPU reference — same algorithm, same inputs copied to CPU.
    feats_c = feats_m.detach().cpu().contiguous()
    coords_c = coords_m.detach().cpu().contiguous()
    weight_c = weight_m.detach().cpu().contiguous()
    bias_c = bias_m.detach().cpu().contiguous()
    grad_out_c = grad_out_m.detach().cpu().contiguous()
    _, gi_c, gw_c, gb_c = _run(algo, feats_c, coords_c, shape, weight_c, bias_c, grad_out_c)

    tol = SMOKE_TOL[dtype]
    assert torch.allclose(gi_cpu, gi_c.float(), **tol), f"grad_input MPS vs CPU mismatch"
    assert torch.allclose(gw_cpu, gw_c.float(), **tol), f"grad_weight MPS vs CPU mismatch"
    assert torch.allclose(gb_cpu, gb_c.float(), **tol), f"grad_bias MPS vs CPU mismatch"
