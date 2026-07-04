"""Differentiable nvdiffrast renderer for the bear 4D pipeline.

Mirrors AniGen ``MeshRenderer`` projection math exactly (normalized OpenCV intrinsics ->
``intrinsics_to_projection`` -> ``proj @ extrinsics``), so cameras built with
``geom.look_at_extrinsics`` + ``geom.fov_to_intrinsics_normalized`` behave identically.

All outputs use **top-left image origin** (row 0 = top, y grows downward), matching PIL /
DAVIS images: the raw nvdiffrast (OpenGL bottom-left) buffer is flipped vertically.

Provides:
    render_silhouette : antialiased, differentiable mask [H,W] in [0,1]
    render_color      : per-vertex-color image [H,W,3] + mask (for previews / output)
    render_flow       : predicted 2D pixel flow [H,W,2] (top-left px) from prev->cur geometry,
                        differentiable w.r.t. the *current* vertices; coverage mask from prev.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import torch
import nvdiffrast.torch as dr

from geometry import intrinsics_to_projection


class Renderer:
    def __init__(self, device="cuda"):
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)

    # -- internals -------------------------------------------------------- #
    def _full_proj(self, extrinsics, intrinsics, near, far):
        persp = intrinsics_to_projection(intrinsics, near, far)      # [4,4]
        return persp @ extrinsics                                    # [4,4]

    def _clip(self, verts, full_proj):
        """verts [V,3] world -> clip [1,V,4]."""
        vh = torch.cat([verts, torch.ones_like(verts[..., :1])], dim=-1)  # [V,4]
        clip = vh @ full_proj.T
        return clip.unsqueeze(0)

    @staticmethod
    def _clip_to_pixels(clip, H, W):
        """clip [1,V,4] -> pixel coords [1,V,2], top-left origin (x right, y down).

        Calibrated to nvdiffrast's *native* raster output (no vertical flip): a point that
        is physically up (small ndc_y) lands on a small row index (top). Calibrated empirically (native raster = top-left).
        """
        ndc = clip[..., :2] / clip[..., 3:4].clamp(min=1e-8)
        px = (0.5 + 0.5 * ndc[..., 0]) * W
        py = (0.5 + 0.5 * ndc[..., 1]) * H
        return torch.stack([px, py], dim=-1)

    # -- silhouette ------------------------------------------------------- #
    def render_silhouette(self, verts, faces_int, extrinsics, intrinsics, H, W,
                          near=0.01, far=100.0, ssaa=1):
        full = self._full_proj(extrinsics, intrinsics, near, far)
        clip = self._clip(verts, full)
        rH, rW = H * ssaa, W * ssaa
        rast, _ = dr.rasterize(self.glctx, clip, faces_int, (rH, rW))
        hard = (rast[..., -1:] > 0).float()
        mask = dr.antialias(hard, rast, clip, faces_int)            # [1,rH,rW,1] differentiable
        mask = mask[0, ..., 0]                                      # native raster = top-left origin
        if ssaa > 1:
            mask = torch.nn.functional.interpolate(
                mask[None, None], size=(H, W), mode="bilinear",
                align_corners=False, antialias=True)[0, 0]
        return mask

    # -- color (per-vertex) ---------------------------------------------- #
    def render_color(self, verts, faces_int, vert_colors, extrinsics, intrinsics, H, W,
                     near=0.01, far=100.0, ssaa=2, bg=1.0):
        full = self._full_proj(extrinsics, intrinsics, near, far)
        clip = self._clip(verts, full)
        rH, rW = H * ssaa, W * ssaa
        rast, _ = dr.rasterize(self.glctx, clip, faces_int, (rH, rW))
        col, _ = dr.interpolate(vert_colors.unsqueeze(0).contiguous(), rast, faces_int)  # [1,rH,rW,3]
        col = dr.antialias(col, rast, clip, faces_int)
        hard = (rast[..., -1:] > 0).float()
        alpha = dr.antialias(hard, rast, clip, faces_int)
        img = col * alpha + bg * (1.0 - alpha)
        img = img[0]                                                # [rH,rW,3] native = top-left
        alpha = alpha[0, ..., 0]
        if ssaa > 1:
            img = torch.nn.functional.interpolate(
                img.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                align_corners=False, antialias=True)[0].permute(1, 2, 0)
            alpha = torch.nn.functional.interpolate(
                alpha[None, None], size=(H, W), mode="bilinear",
                align_corners=False, antialias=True)[0, 0]
        return img.clamp(0, 1), alpha.clamp(0, 1)

    # -- flow ------------------------------------------------------------- #
    def render_flow(self, verts_prev, verts_cur, faces_int, extrinsics, intrinsics, H, W,
                    near=0.01, far=100.0):
        """Predicted pixel flow prev->cur [H,W,2] (top-left px) + coverage mask [H,W].

        Rasterizes the mesh at the *previous* frame (fixed geometry) and interpolates the
        per-vertex pixel displacement (cur - prev). Gradients flow to verts_cur only.
        """
        full = self._full_proj(extrinsics, intrinsics, near, far)
        clip_prev = self._clip(verts_prev, full)
        clip_cur = self._clip(verts_cur, full)
        px_prev = self._clip_to_pixels(clip_prev, H, W)             # [1,V,2]
        px_cur = self._clip_to_pixels(clip_cur, H, W)
        disp = (px_cur - px_prev).contiguous()                      # [1,V,2]
        rast, _ = dr.rasterize(self.glctx, clip_prev, faces_int, (H, W))
        flow, _ = dr.interpolate(disp, rast, faces_int)             # [1,H,W,2]
        cover = (rast[..., -1] > 0).float()                         # [1,H,W]
        flow = flow[0]                                              # native = top-left
        cover = cover[0]
        return flow, cover


def to_uint8(img: torch.Tensor):
    """[H,W,3] float [0,1] -> HxWx3 uint8 numpy."""
    return (img.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
