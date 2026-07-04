"""Stage 3: refine the object's rigid pose on DAVIS frame 0 with nvdiffrast.

Fixed full-frame camera (from VGGT).  Optimise global {scale, rotation(6D), translation} of
the *rest* mesh so its differentiable silhouette matches the DAVIS frame-0 mask.  Coarse-to-
fine blur on the silhouette loss widens the basin of attraction.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import os

import numpy as np
import torch
from PIL import Image

from geometry import rot6d_to_matrix, identity_rot6d, apply_similarity
from renderer import Renderer, to_uint8
from davis import load_davis, davis_paths
from fit_utils import build_fit_camera, silhouette_loss, mask_iou, init_similarity


def overlay(frame_rgb, pred_mask, target_mask):
    """frame_rgb [H,W,3] uint8; draw pred (green) & target (red) mask boundaries."""
    import cv2
    img = frame_rgb.copy()
    for m, col in [(target_mask, (255, 0, 0)), (pred_mask, (0, 255, 0))]:
        mm = (m > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--vggt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=540)
    ap.add_argument("--fov_x", type=float, default=50.0)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    dev = "cuda"
    Rd = f"results/{args.seq}"
    args.rig = args.rig or f"{Rd}/rig.npz"
    args.vggt = args.vggt or f"{Rd}/vggt_pose.npz"
    args.out = args.out or f"{Rd}/pose0.npz"
    frames_dir, ann_dir = davis_paths(args.seq)

    d = np.load(args.rig)
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32)

    vg = np.load(args.vggt)
    frames, masks, names, _ = load_davis(frames_dir, ann_dir, H=args.H, W=args.W, n_frames=1)
    target = torch.tensor(masks[0], device=dev, dtype=torch.float32)
    frame0 = (frames[0] * 255).astype(np.uint8)

    E, K, info = build_fit_camera(vg["R_real_canon"], vg["cam_dir"], args.W, args.H,
                                  fov_x_deg=args.fov_x, radius=args.radius, device=dev)
    r = Renderer(dev)

    s0, tg0 = init_similarity(r, verts, faces, E, K, args.W, args.H, target, info, dev)
    print(f"init: scale={s0:.3f} tg={tg0.cpu().numpy().round(3).tolist()}")

    log_s = torch.tensor(np.log(s0), device=dev, requires_grad=True)
    r6 = identity_rot6d(1, device=dev)[0].clone().requires_grad_(True)
    tg = tg0.clone().requires_grad_(True)

    opt = torch.optim.Adam([{"params": [r6], "lr": 0.02},
                            {"params": [log_s], "lr": 0.01},
                            {"params": [tg], "lr": 0.01}])

    for it in range(args.iters):
        # coarse-to-fine blur
        frac = it / max(1, args.iters - 1)
        sigma = max(1.0, 6.0 * (1 - frac))
        ksize = int(sigma * 4) | 1
        Rg = rot6d_to_matrix(r6)
        s = torch.exp(log_s)
        vw = apply_similarity(verts, s, Rg, tg)
        pred = r.render_silhouette(vw, faces, E, K, args.H, args.W, ssaa=1)
        loss = silhouette_loss(pred, target, w_iou=1.0, w_l2=2.0, blur=True, ksize=ksize, sigma=sigma)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"  it {it:4d}  loss={loss.item():.4f}  iou={mask_iou(pred, target):.3f}  "
                  f"scale={s.item():.3f}")

    with torch.no_grad():
        Rg = rot6d_to_matrix(r6)
        s = torch.exp(log_s)
        vw = apply_similarity(verts, s, Rg, tg)
        pred = r.render_silhouette(vw, faces, E, K, args.H, args.W, ssaa=2)
        final_iou = mask_iou(pred, target)
    print(f"FINAL frame-0 IoU = {final_iou:.3f}")

    ov = overlay(frame0, pred.cpu().numpy(), masks[0])
    Image.fromarray(ov).save(os.path.join(os.path.dirname(args.out), "pose0_overlay.png"))

    np.savez(args.out,
             scale=s.item(),
             rot6d=r6.detach().cpu().numpy().astype(np.float32),
             Rg=Rg.detach().cpu().numpy().astype(np.float32),
             tg=tg.detach().cpu().numpy().astype(np.float32),
             E_fit=E.cpu().numpy().astype(np.float32),
             K_norm=K.cpu().numpy().astype(np.float32),
             W=args.W, H=args.H, fov_x=info["fov_x"], radius=args.radius,
             iou=final_iou)
    print(f"saved -> {args.out}  (overlay: pose0_overlay.png)")


if __name__ == "__main__":
    main()
