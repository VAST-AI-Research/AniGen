"""Rasterizer smoke tests for the Apple Silicon (MPS) inference port (Task 10).

These VERIFY the rasterization paths that Task 4 re-pointed from nvdiffrast to
mtldiffrast, and surface the spec's top risk: whether utils3d's CPU
rasterization (used for texture-bake / hole-fill visibility in
``anigen/utils/postprocessing_utils.py``) is viable on Mac.

Outcomes (see Step 3 below):
  * mtldiffrast renderer construct + trivial rasterize  -> PASS on MPS.
  * utils3d CPU rasterization                           -> NOT VIABLE (xfail).
"""
import torch
import pytest


# ---------------------------------------------------------------------------
# Step 1: the real AniGen renderer constructs on MPS using mtldiffrast.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_mesh_renderer_constructs_on_mps():
    # MeshRenderer.__init__(rendering_options={}, device='cuda'); on a non-cuda
    # device it builds self.glctx = mtldiffrast.torch.MtlRasterizeContext(device).
    from anigen.renderers.mesh_renderer import MeshRenderer

    r = MeshRenderer(device="mps")
    assert r.glctx is not None
    assert r.device == "mps"


# ---------------------------------------------------------------------------
# Step 2: prove the Metal rasterizer actually runs (not just constructs).
# Rasterize ONE center-covering triangle directly through mtldiffrast.torch.
# Real API (inspected):
#   MtlRasterizeContext(self, device=None)
#   rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True)
#     -> (rast, rast_db);  rast is [B, H, W, 4] = (u, v, z, triangle_id + 1)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_mtldiffrast_rasterize_single_triangle():
    import mtldiffrast.torch as dr

    ctx = dr.MtlRasterizeContext(device="mps")
    # clip-space verts [x, y, z, w] for one triangle covering the center.
    pos = torch.tensor(
        [[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0.0, 0.5, 0, 1]]],
        device="mps",
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], device="mps", dtype=torch.int32)

    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])

    # rast is [1, H, W, 4] = (u, v, z, triangle_id + 1).
    assert rast.shape == (1, 64, 64, 4)
    assert torch.isfinite(rast).all()
    # A center-covering triangle MUST produce at least one covered pixel
    # (triangle_id channel > 0). This is the assertion that matters.
    assert (rast[..., 3] > 0).any()


# ---------------------------------------------------------------------------
# Step 3 (THE RISK CHECK): does utils3d CPU rasterization work on Mac?
#
# OUTCOME: B -- NOT VIABLE.
#
# ``anigen/utils/postprocessing_utils.py`` calls
#   rastctx = utils3d.torch.RastContext(backend=_rast_backend())   # 'cpu' on Mac
#   utils3d.torch.rasterize_triangle_faces(rastctx, verts[None], faces, ...)
#
# But the INSTALLED utils3d (1.3) cannot support this on Mac:
#
#   1. ``utils3d/torch/rasterization.py`` unconditionally does
#      ``import nvdiffrast.torch as dr`` at module top -- no fallback. Merely
#      touching ``utils3d.torch.RastContext`` raises
#      ``ModuleNotFoundError: No module named 'nvdiffrast'`` on a clean Mac env.
#   2. ``RastContext`` only accepts backend in {'gl', 'cuda'}; ``backend='cpu'``
#      raises ``ValueError('Unknown backend: cpu')``. There is NO CPU backend.
#   3. ``rasterize_triangle_faces`` is not even a top-level export of this
#      utils3d version (the real function is ``rasterize_triangles``).
#
# Task-11 implication: texture-bake / hole-fill visibility via utils3d CPU is
# NOT viable. Texture baking must be gated behind a --no_texture path so that
# mesh + skeleton + skin still export. (The --no_texture flag is Task 11's job,
# not added here.)
#
# This test PROVES the failure precisely instead of pretending it works.
# Note: tests/mps/conftest.py registers a lazy stub for ``nvdiffrast`` so that
# ``import anigen.*`` can complete on a clean Mac. That stub would mask the
# ModuleNotFoundError above. We therefore detect EITHER the real
# ModuleNotFoundError (no stub) OR the no-CPU-backend ValueError (stub active);
# both demonstrate utils3d CPU rasterization is non-functional on Mac.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason=(
        "OUTCOME B: utils3d 1.3 CPU rasterization is NOT viable on Mac. "
        "utils3d/torch/rasterization.py hard-imports nvdiffrast and RastContext "
        "has no 'cpu' backend. Task 11 must gate texture baking behind "
        "--no_texture so mesh+skeleton+skin still export."
    ),
    strict=True,
    raises=(ModuleNotFoundError, ValueError, AttributeError),
)
def test_utils3d_cpu_rasterization_viability():
    import utils3d

    # Mirrors postprocessing_utils.py: build a CPU rast context, then rasterize
    # a tiny triangle. EITHER step raises on Mac (see module docstring above).
    rastctx = utils3d.torch.RastContext(backend="cpu")  # ValueError / ModuleNotFoundError

    verts = torch.tensor(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    view = utils3d.torch.view_look_at(
        torch.tensor([0.0, 0.0, 2.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
    )
    projection = utils3d.torch.perspective_from_fov_xy(
        torch.deg2rad(torch.tensor(40.0)), torch.deg2rad(torch.tensor(40.0)), 1, 3
    )
    buffers = utils3d.torch.rasterize_triangle_faces(  # AttributeError: not exported
        rastctx, verts[None], faces, 64, 64, view=view, projection=projection
    )
    # If utils3d ever gains a working CPU backend, these assertions guard it.
    assert torch.isfinite(buffers["mask"][0]).all()
    assert (buffers["mask"][0] > 0).any()
