"""Shared utilities for pose refinement and skeleton motion fitting.

Fixed full-frame fitting camera
-------------------------------
Camera rotation = VGGT-estimated ``R_real_canon`` (so the rest mesh, unrotated, already
matches the real bear orientation), camera centre placed at ``radius * cam_dir`` looking at
the origin, with full-frame (16:9) intrinsics from an assumed horizontal FOV.  The object is
positioned/scaled/animated in the *canonical world*; its apparent translation across the
DAVIS frame is absorbed by the per-frame root translation ``tg``.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import numpy as np
import torch
import torch.nn.functional as F

from geometry import fov_to_intrinsics_normalized


def build_fit_camera(R_real_canon, cam_dir, W, H, fov_x_deg=50.0, radius=3.0, device="cuda"):
    R_fit = torch.as_tensor(R_real_canon, device=device, dtype=torch.float32)
    cam_dir = torch.as_tensor(cam_dir, device=device, dtype=torch.float32)
    C_fit = radius * cam_dir
    t_fit = -R_fit @ C_fit
    E = torch.eye(4, device=device, dtype=torch.float32)
    E[:3, :3] = R_fit
    E[:3, 3] = t_fit
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = 2 * np.arctan(np.tan(fov_x / 2) * H / W)
    K = fov_to_intrinsics_normalized(fov_x, fov_y, device=device)
    info = dict(radius=float(radius), fov_x=float(fov_x), fov_y=float(fov_y),
                fx_px=float(K[0, 0].item() * W), fy_px=float(K[1, 1].item() * H),
                cx_px=0.5 * W, cy_px=0.5 * H)
    return E, K, info


def pick_vertex_colors(d):
    """Vertex colours to render: the per-vertex texture-optimised colours if coverage >= threshold, else
    (low coverage) the global-colormap colours (`vertex_colors_global`) if present, else the raw original."""
    files = getattr(d, "files", list(d.keys()))
    cov = float(d["tex_coverage"]) if "tex_coverage" in files else 1.0
    th = float(d["tex_cov_thresh"]) if "tex_cov_thresh" in files else 0.70
    if cov < th:
        if "vertex_colors_global" in files:
            return d["vertex_colors_global"]
        if "vertex_colors_orig" in files:
            return d["vertex_colors_orig"]
    return d["vertex_colors"]


def bilateral_time_smooth(x, sigma_t, k_v=1.5):
    """Edge-preserving temporal smoothing of a per-frame signal x[T, ...].

    Each frame is a weighted average of its temporal neighbours, with weight = Gaussian-in-time *
    similarity kernel on the per-frame change.  Slow motion (a smooth walk) is smoothed away, but a
    FAST change (a quick turn) is preserved because frames across it are dissimilar -> low weight.
    The change scale auto-adapts to k_v * median(|Δx|), so it works across sequences.  Returns np[T,...].
    """
    x = np.asarray(x, np.float64)
    if sigma_t <= 0 or len(x) < 3:
        return x
    T = x.shape[0]; xf = x.reshape(T, -1)
    vel = np.linalg.norm(np.diff(xf, axis=0), axis=1)
    sv = max(1e-9, k_v * float(np.median(vel)))
    r = int(3 * sigma_t) + 1
    out = xf.copy()
    for t in range(T):
        lo, hi = max(0, t - r), min(T, t + r + 1)
        dt = np.arange(lo, hi) - t
        wt = np.exp(-dt ** 2 / (2 * sigma_t ** 2))
        dv = np.linalg.norm(xf[lo:hi] - xf[t], axis=1)
        w = wt * np.exp(-(dv / sv) ** 2 / 2)
        w /= w.sum()
        out[t] = (w[:, None] * xf[lo:hi]).sum(0)
    return out.reshape(x.shape)


def gaussian_blur(x, ksize=11, sigma=3.0):
    """Blur a [H,W] or [1,1,H,W] tensor with a separable Gaussian."""
    if x.dim() == 2:
        x = x[None, None]
    k = torch.arange(ksize, device=x.device, dtype=x.dtype) - (ksize - 1) / 2
    g = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    gx = g.view(1, 1, 1, ksize)
    gy = g.view(1, 1, ksize, 1)
    x = F.conv2d(x, gx, padding=(0, ksize // 2))
    x = F.conv2d(x, gy, padding=(ksize // 2, 0))
    return x[0, 0]


def soft_iou_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = (pred + target - pred * target).sum()
    return 1.0 - (inter + eps) / (union + eps)


def silhouette_loss(pred, target, w_iou=1.0, w_l2=1.0, blur=True, ksize=15, sigma=4.0):
    """Combined soft-IoU + (optionally blurred) L2 between rendered and target masks [H,W]."""
    loss = w_iou * soft_iou_loss(pred, target)
    if w_l2 > 0:
        if blur:
            p = gaussian_blur(pred, ksize, sigma)
            t = gaussian_blur(target, ksize, sigma)
        else:
            p, t = pred, target
        loss = loss + w_l2 * ((p - t) ** 2).mean()
    return loss


def mask_iou(pred, target, thr=0.5):
    p = (pred > thr).float()
    t = (target > thr).float()
    inter = (p * t).sum()
    union = (p + t - p * t).sum().clamp_min(1.0)
    return (inter / union).item()


def mask_centroid_area_t(mask):
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if len(xs) == 0:
        H, W = mask.shape
        return torch.tensor([W / 2.0, H / 2.0], device=mask.device), 0.0
    return torch.stack([xs.float().mean(), ys.float().mean()]), float(len(xs))


def init_similarity(renderer, verts, faces, E, K, W, H, target_mask, info, device="cuda"):
    """Initialise global scale + translation from mask area & centroid (rotation = identity)."""
    with torch.no_grad():
        m0 = renderer.render_silhouette(verts, faces, E, K, H, W)
        a_r = m0.sum().clamp_min(1.0)
        a_t = (target_mask > 0.5).float().sum().clamp_min(1.0)
        s = torch.sqrt(a_t / a_r).item()

        # scaled render -> centroid, then shift to target centroid
        vs = s * verts
        ms = renderer.render_silhouette(vs, faces, E, K, H, W)
        pr, _ = mask_centroid_area_t(ms)
        pt, _ = mask_centroid_area_t(target_mask)
        dpix = pt - pr
        Z = info["radius"]
        dxc = Z * dpix[0].item() / info["fx_px"]
        dyc = Z * dpix[1].item() / info["fy_px"]
        R_fit = E[:3, :3]
        tg = (R_fit.T @ torch.tensor([dxc, dyc, 0.0], device=device))
    return s, tg
