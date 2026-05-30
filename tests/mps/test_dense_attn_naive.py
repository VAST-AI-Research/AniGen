"""Parity guard for the DENSE attention `naive` backend.

AniGen's dense attention (`anigen/modules/attention/full_attn.py`) already ships
a real fp32 matmul+softmax `naive` path (`_naive_sdpa`, dispatched from the public
`scaled_dot_product_attention` when `ATTN_BACKEND=naive`). On Apple Silicon we
force this backend because MPS fused SDPA is banned. These tests lock in that the
naive path matches an independent explicit reference attention, on CPU and MPS.

The public entry point is `scaled_dot_product_attention` (per `__all__`). Its
3-argument form expects `q`, `k`, `v` each shaped `[N, L, H, C]` and returns
`[N, L, H, C]` (see the file's `@overload` docstrings).

`anigen_mps` (imported via conftest) sets `ATTN_BACKEND=naive` before the package
is imported, so the backend is already `naive`; we re-assert the env var and
reload the module defensively so this test is correct in isolation too.
"""
import os
import math
import importlib

import torch


def test_naive_matches_reference_sdpa():
    os.environ["ATTN_BACKEND"] = "naive"
    import anigen.modules.attention.full_attn as fa
    importlib.reload(fa)
    assert fa.BACKEND == "naive", f"expected naive backend, got {fa.BACKEND}"

    N, L, H, C = 2, 64, 4, 32
    q = torch.randn(N, L, H, C)
    k = torch.randn(N, L, H, C)
    v = torch.randn(N, L, H, C)
    out = fa.scaled_dot_product_attention(q, k, v)

    # reference: explicit fp32 attention, independent of the code under test
    qr, kr, vr = (t.permute(0, 2, 1, 3) for t in (q, k, v))   # [N, H, L, C]
    w = torch.softmax(qr @ kr.transpose(-2, -1) / math.sqrt(C), dim=-1)
    ref = (w @ vr).permute(0, 2, 1, 3)                        # [N, L, H, C]

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_naive_matches_reference_on_mps():
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS unavailable")
    os.environ["ATTN_BACKEND"] = "naive"
    import anigen.modules.attention.full_attn as fa
    importlib.reload(fa)
    assert fa.BACKEND == "naive", f"expected naive backend, got {fa.BACKEND}"

    N, L, H, C = 2, 64, 4, 32
    q = torch.randn(N, L, H, C, device="mps")
    k = torch.randn(N, L, H, C, device="mps")
    v = torch.randn(N, L, H, C, device="mps")
    out = fa.scaled_dot_product_attention(q, k, v).cpu()

    # reference computed on CPU from the same inputs
    qc, kc, vc = q.cpu(), k.cpu(), v.cpu()
    qr, kr, vr = (t.permute(0, 2, 1, 3) for t in (qc, kc, vc))   # [N, H, L, C]
    w = torch.softmax(qr @ kr.transpose(-2, -1) / math.sqrt(C), dim=-1)
    ref = (w @ vr).permute(0, 2, 1, 3)                           # [N, L, H, C]

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
