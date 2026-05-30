"""Rasterizer smoke tests for the Apple Silicon (MPS) inference port (Task 10).

These VERIFY the rasterization paths that Task 4 re-pointed from nvdiffrast to
mtldiffrast, and surface the spec's top risk: whether utils3d's CPU
rasterization (used for texture-bake / hole-fill visibility in
``anigen/utils/postprocessing_utils.py``) is viable on Mac.

Outcomes (see Step 3 below):
  * mtldiffrast renderer construct + trivial rasterize  -> PASS on MPS.
  * utils3d rasterization via nvdiffrast->mtldiffrast alias -> PASS on Metal (Task 11a).
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
# Step 3 (THE RISK CHECK): does utils3d rasterization work on Mac?
#
# OUTCOME: VIABLE on Metal via the nvdiffrast->mtldiffrast alias (Task 11a).
#
# ``anigen/utils/postprocessing_utils.py`` (hole-fill visibility / texture bake)
# calls
#   rastctx = utils3d.torch.RastContext(backend=_rast_backend())   # 'cuda' on Mac
#   utils3d.torch.rasterize_triangle_faces(rastctx, verts[None], faces, ...)
#
# utils3d 0.0.2's ``utils3d/torch/rasterization.py`` hard-imports
# ``nvdiffrast.torch`` at module top and ``RastContext`` instantiates
# ``dr.RasterizeGLContext`` / ``dr.RasterizeCudaContext``. There is NO 'cpu'
# backend (backend in {'gl','cuda'} only).
#
# ``anigen_mps.install_nvdiffrast_alias()`` (run on ``import anigen_mps``)
# builds a synthetic ``nvdiffrast.torch`` backed by mtldiffrast and aliases both
# context-class names to ``MtlRasterizeContext``. So the hard import succeeds and
# the rasterizer runs on Metal. ``_rast_backend()`` returns 'cuda' on Mac (which
# now maps to the Metal context); 'gl' would too, since both aliases point at
# MtlRasterizeContext. We exercise the 'cuda' path here because that is exactly
# what postprocessing_utils.py requests.
#
# This test runs the SAME arg shapes postprocessing_utils.py uses
# (perspective_from_fov_xy/view_look_at, verts [1,V,3], int faces, square res)
# and asserts a finite buffer with at least one covered pixel.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_utils3d_rasterization_on_metal():
    import anigen_mps  # noqa: F401  -- installs the nvdiffrast->mtldiffrast alias
    import utils3d.torch as u3

    dev = "mps"
    # 'cuda' is what anigen.utils.postprocessing_utils._rast_backend() requests on
    # Mac; the alias routes it to mtldiffrast's MtlRasterizeContext (Metal).
    rastctx = u3.RastContext(backend="cuda", device=dev)

    verts = torch.tensor(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
        dtype=torch.float32,
        device=dev,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=dev)
    view = u3.view_look_at(
        torch.tensor([0.0, 0.0, 2.0], device=dev),
        torch.tensor([0.0, 0.0, 0.0], device=dev),
        torch.tensor([0.0, 1.0, 0.0], device=dev),
    )
    projection = u3.perspective_from_fov_xy(
        torch.deg2rad(torch.tensor(40.0, device=dev)),
        torch.deg2rad(torch.tensor(40.0, device=dev)),
        1,
        3,
    )
    buffers = u3.rasterize_triangle_faces(
        rastctx, verts[None], faces, 64, 64, view=view, projection=projection
    )

    mask = buffers["mask"][0]
    face_id = buffers["face_id"][0]
    assert torch.isfinite(mask).all()
    assert torch.isfinite(face_id).all()
    # A center-covering triangle MUST yield at least one covered pixel.
    assert (mask > 0).any()
