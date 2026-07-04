"""Visualize the SpatialTracker V2 3D tracking supervision used in the motion fit.

Produces:
  tracks_2d_overlay.mp4/.gif + sheet : per DAVIS frame, the SpatialTracker track TARGETS
      (filled dots) vs the fitted mesh's PREDICTED reprojection (rings) + residual lines +
      motion trails.  This is exactly the 2D reprojection term the optimizer minimizes.
  tracks_3d_scatter.png              : the raw 3D world tracks (bound subset) colored by
      camera depth, plus time-colored 3D trajectories -> the signal the 3D term uses.
  tracks_3d_novelview.mp4/.gif       : the animated mesh from a novel 3/4 view with the bound
      tracked points riding on the surface + trails -> shows the 3D binding.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import os

import numpy as np
import torch
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from geometry import (rot6d_to_matrix, apply_similarity, Skeleton,
                   look_at_extrinsics, fov_to_intrinsics_normalized, intrinsics_to_projection)
from renderer import Renderer, to_uint8
from tracks import TrackSupervisor
from davis import load_davis, davis_paths


def track_colors(uv0, H):
    """Per-track RGB by frame-0 position (spatially-coherent rainbow) -> readable trails."""
    hue = (uv0[:, 1] / H).clip(0, 1)                      # top->bottom hue sweep
    hsv = np.stack([hue * 179, np.full_like(hue, 210), np.full_like(hue, 255)], -1).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]  # [N,3] uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--tracks", "--spatrack", dest="tracks", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--trail", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda"
    Rd = f"results/{args.seq}"
    args.rig = args.rig or f"{Rd}/rig.npz"
    args.motion = args.motion or f"{Rd}/motion.npz"
    args.tracks = args.tracks or f"{Rd}/cotracker.npz"
    args.outdir = args.outdir or f"{Rd}/renders"
    frames_dir, ann_dir = davis_paths(args.seq)
    os.makedirs(args.outdir, exist_ok=True)

    rig = np.load(args.rig)
    verts = torch.tensor(rig["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(rig["faces"], device=dev, dtype=torch.int32)
    colors = torch.tensor(rig["vertex_colors"], device=dev, dtype=torch.float32)
    weights = torch.tensor(rig["skin_weights"], device=dev, dtype=torch.float32)
    sk = Skeleton(rig["joints"], rig["parents"], device=dev)

    m = np.load(args.motion, allow_pickle=True)
    bone6 = torch.tensor(m["bone6"], device=dev, dtype=torch.float32)
    r6 = torch.tensor(m["r6"], device=dev, dtype=torch.float32)
    tg = torch.tensor(m["tg"], device=dev, dtype=torch.float32)
    s = float(m["scale"]); W, H = int(m["W"]), int(m["H"])
    E = torch.tensor(m["E_fit"], device=dev, dtype=torch.float32)
    K = torch.tensor(m["K_norm"], device=dev, dtype=torch.float32)
    T = bone6.shape[0]

    def world_verts(t, root=True):
        vc = sk.lbs(verts, weights, rot6d_to_matrix(bone6[t]))
        tt = tg[t] if root else torch.zeros(3, device=dev)
        return apply_similarity(vc, s, rot6d_to_matrix(r6[t]), tt)

    r = Renderer(dev)
    ts = TrackSupervisor(args.tracks, E, K, W, H, r.glctx, device=dev)
    nb, resid = ts.bind(world_verts(0).detach(), faces)
    bnd = ts.bnd
    print(f"visualizing {nb} bound tracks (Sim3 residual {resid:.4f})")

    # precompute targets (SpatialTracker 2D) + predictions (mesh reprojection) per frame
    frames_np, _, _, _ = load_davis(frames_dir, ann_dir, H=H, W=W, n_frames=T)
    full = intrinsics_to_projection(K, ts.near, ts.far) @ E
    tgt_uv = ts.uv[:, bnd].cpu().numpy()                  # [T,Nb,2]
    valid = ts.valid[:, bnd].cpu().numpy()               # [T,Nb]
    pred_uv = np.zeros_like(tgt_uv)
    bound_world = []
    with torch.no_grad():
        for t in range(T):
            vw = world_verts(t)
            pts = ts._bound_points(vw)                    # [Nb,3] world
            bound_world.append(pts.cpu().numpy())
            vh = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)
            clip = vh @ full.T
            ndc = clip[..., :2] / clip[..., 3:4].clamp(min=1e-6)
            pred_uv[t, :, 0] = ((0.5 + 0.5 * ndc[..., 0]) * W).cpu().numpy()
            pred_uv[t, :, 1] = ((0.5 + 0.5 * ndc[..., 1]) * H).cpu().numpy()
    bound_world = np.stack(bound_world, 0)               # [T,Nb,3]
    col = track_colors(tgt_uv[0], H)                      # [Nb,3]

    # ---------- (1) 2D supervision overlay ----------
    vid2d = []
    for t in range(T):
        img = (frames_np[t] * 255).astype(np.uint8).copy()
        img = (img * 0.55).astype(np.uint8)              # dim background for contrast
        for i in range(len(bnd)):
            if not valid[t, i]:
                continue
            c = tuple(int(x) for x in col[i])
            # trail of targets
            for k in range(max(1, t - args.trail), t + 1):
                if valid[k, i] and valid[k - 1, i]:
                    p0 = tuple(np.round(tgt_uv[k - 1, i]).astype(int))
                    p1 = tuple(np.round(tgt_uv[k, i]).astype(int))
                    cv2.line(img, p0, p1, c, 1, cv2.LINE_AA)
            tp = tuple(np.round(tgt_uv[t, i]).astype(int))
            pp = tuple(np.round(pred_uv[t, i]).astype(int))
            cv2.line(img, tp, pp, (255, 255, 0), 1, cv2.LINE_AA)     # residual (yellow)
            cv2.circle(img, tp, 3, c, -1, cv2.LINE_AA)              # target (filled)
            cv2.circle(img, pp, 4, (255, 255, 255), 1, cv2.LINE_AA)  # prediction (white ring)
        err = np.linalg.norm((pred_uv[t] - tgt_uv[t])[valid[t]], axis=-1).mean() if valid[t].any() else 0
        cv2.putText(img, f"{ts.source} target (dot)  mesh pred (ring)  residual (yellow) | frame {t:02d}  err {err:4.1f}px",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vid2d.append(img)
    imageio.mimsave(os.path.join(args.outdir, "tracks_2d_overlay.mp4"), vid2d, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimsave(os.path.join(args.outdir, "tracks_2d_overlay.gif"),
                    [np.asarray(Image.fromarray(f).resize((W // 2, H // 2))) for f in vid2d], fps=args.fps, loop=0)
    idx = np.linspace(0, T - 1, 6).astype(int)
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(vid2d[i]).resize((360, 203))) for i in idx], 1)) \
        .save(os.path.join(args.outdir, "tracks_2d_sheet.png"))
    print("  wrote tracks_2d_overlay.mp4/.gif + sheet")

    # ---------- (2) 3D scatter of the raw world tracks (SpatialTracker only; needs 3D) ----------
    if ts.has3d:
        wt = np.load(args.tracks)["world_tracks"][:, bnd.cpu().numpy()]   # [T,Nb,3]
        camz = ts.cam[:, bnd][0, :, 2].cpu().numpy()                      # frame-0 camera depth
        fig = plt.figure(figsize=(13, 5.5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        p = wt[0]
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=camz, cmap="viridis", s=8)
        ax.set_title("3D tracks @ frame0 (colored by camera depth)")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="camera depth")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        sub = np.linspace(0, len(bnd) - 1, min(60, len(bnd))).astype(int)
        for j in sub:
            tr = wt[:, j]
            ax2.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=0.8, alpha=0.7, color=plt.cm.turbo(j / len(bnd)))
        ax2.set_title("Per-track 3D trajectories over the video")
        for a in (ax, ax2):
            a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z")
            try:
                a.set_box_aspect((np.ptp(wt[..., 0]), np.ptp(wt[..., 1]), np.ptp(wt[..., 2])))
            except Exception:
                pass
            a.view_init(elev=-70, azim=-90)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "tracks_3d_scatter.png"), dpi=120)
        plt.close(fig)
        print("  wrote tracks_3d_scatter.png")
    else:
        print("  (3D scatter skipped: CoTracker3 is 2D-only)")

    # ---------- (3) novel-view: tracked points riding the animated mesh ----------
    Kc = fov_to_intrinsics_normalized(np.deg2rad(40), np.deg2rad(40), device=dev)
    res = 640
    with torch.no_grad():
        v0c = world_verts(0, root=False)
        extent = (v0c.max(0).values - v0c.min(0).values).max().item()
    radius = 2.4 * extent
    az, el = np.deg2rad(90 + 55), np.deg2rad(18)         # 3/4 novel view
    eye = radius * torch.tensor([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)],
                                device=dev, dtype=torch.float32)
    Ec = look_at_extrinsics(eye, [0, 0, 0], up=[0, 0, 1], device=dev)
    fullc = intrinsics_to_projection(Kc, 0.01, 100.0) @ Ec
    vid3d = []
    pix_hist = np.zeros((T, len(bnd), 2))
    with torch.no_grad():
        for t in range(T):
            vw = world_verts(t, root=False)
            center = vw.mean(0)
            vwc = vw - center
            img, _ = r.render_color(vwc, faces, colors, Ec, Kc, res, res, ssaa=2, bg=0.92)
            frame = to_uint8(img).copy()
            pts = ts._bound_points(vw) - center
            vh = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)
            clip = vh @ fullc.T
            ndc = (clip[..., :2] / clip[..., 3:4].clamp(min=1e-6)).cpu().numpy()
            px = (0.5 + 0.5 * ndc[:, 0]) * res
            py = (0.5 + 0.5 * ndc[:, 1]) * res
            pix_hist[t, :, 0] = px; pix_hist[t, :, 1] = py
            for i in range(len(bnd)):
                if not valid[t, i]:
                    continue
                c = tuple(int(x) for x in col[i])
                for k in range(max(1, t - args.trail), t + 1):
                    if valid[k, i] and valid[k - 1, i]:
                        cv2.line(frame, tuple(pix_hist[k - 1, i].round().astype(int)),
                                 tuple(pix_hist[k, i].round().astype(int)), c, 1, cv2.LINE_AA)
                cv2.circle(frame, (int(round(px[i])), int(round(py[i]))), 3, c, -1, cv2.LINE_AA)
            cv2.putText(frame, f"tracked points on 3D mesh (novel view) | frame {t:02d}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            vid3d.append(frame)
    imageio.mimsave(os.path.join(args.outdir, "tracks_3d_novelview.mp4"), vid3d, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimsave(os.path.join(args.outdir, "tracks_3d_novelview.gif"),
                    [np.asarray(Image.fromarray(f).resize((res // 2, res // 2))) for f in vid3d], fps=args.fps, loop=0)
    print("  wrote tracks_3d_novelview.mp4/.gif")
    print(f"all track visualizations saved to {args.outdir}")


if __name__ == "__main__":
    main()
