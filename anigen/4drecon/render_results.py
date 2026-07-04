"""Stage 5: render the fitted 4D bear from the original camera and a cyclic orbit.

* original view : animated mesh (full root motion) rendered from the fixed fitting camera,
                  composited over the DAVIS frames -> shows fit quality on the real video.
* cyclic view   : animated mesh centred at the origin (root translation removed) viewed from
                  a seamless closed-loop orbit  az(t) = az0 + amp*sin(2*pi*t)  (ping-pong),
                  fixed elevation -> a looping novel-view showcase of the articulation.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import os

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

from geometry import (rot6d_to_matrix, apply_similarity, Skeleton,
                   look_at_extrinsics, fov_to_intrinsics_normalized)
from renderer import Renderer, to_uint8
from davis import load_davis, davis_paths


def save_video(frames_u8, path, fps=15, gif=True, gif_scale=0.5):
    imageio.mimsave(path, frames_u8, fps=fps, quality=8, macro_block_size=1)
    if gif:
        gp = os.path.splitext(path)[0] + ".gif"
        small = []
        for f in frames_u8:
            h, w = f.shape[:2]
            small.append(np.asarray(Image.fromarray(f).resize(
                (int(w * gif_scale), int(h * gif_scale)), Image.BILINEAR)))
        imageio.mimsave(gp, small, fps=fps, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cyc_amp", type=float, default=140.0, help="cyclic azimuth swing (deg)")
    ap.add_argument("--cyc_el", type=float, default=20.0, help="cyclic elevation (deg)")
    ap.add_argument("--cyc_res", type=int, default=720)
    args = ap.parse_args()
    dev = "cuda"
    Rd = f"results/{args.seq}"
    args.rig = args.rig or f"{Rd}/rig.npz"
    args.motion = args.motion or f"{Rd}/motion.npz"
    args.outdir = args.outdir or f"{Rd}/renders"
    frames_dir, ann_dir = davis_paths(args.seq)
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.rig)
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32)
    colors = torch.tensor(d["vertex_colors"], device=dev, dtype=torch.float32)
    weights = torch.tensor(d["skin_weights"], device=dev, dtype=torch.float32)
    sk = Skeleton(d["joints"], d["parents"], device=dev)

    m = np.load(args.motion, allow_pickle=True)
    bone6 = torch.tensor(m["bone6"], device=dev, dtype=torch.float32)   # [T,M,6]
    r6 = torch.tensor(m["r6"], device=dev, dtype=torch.float32)         # [T,6]
    tg = torch.tensor(m["tg"], device=dev, dtype=torch.float32)         # [T,3]
    s = float(m["scale"])
    W, H = int(m["W"]), int(m["H"])
    E = torch.tensor(m["E_fit"], device=dev, dtype=torch.float32)
    K = torch.tensor(m["K_norm"], device=dev, dtype=torch.float32)
    T = bone6.shape[0]
    ious = m["iou"]
    print(f"frames={T}  mean IoU={float(np.mean(ious)):.3f}")

    frames_np, _, names, _ = load_davis(frames_dir, ann_dir, H=H, W=W, n_frames=T)
    r = Renderer(dev)

    def world_verts(t, with_root_t=True):
        R_local = rot6d_to_matrix(bone6[t])
        v_can = sk.lbs(verts, weights, R_local)
        t_vec = tg[t] if with_root_t else torch.zeros(3, device=dev)
        return apply_similarity(v_can, s, rot6d_to_matrix(r6[t]), t_vec)

    # ---- original view (composite over DAVIS) ----
    orig_frames = []
    with torch.no_grad():
        for t in range(T):
            vw = world_verts(t, with_root_t=True)
            img, alpha = r.render_color(vw, faces, colors, E, K, H, W, ssaa=2, bg=1.0)
            a = alpha[..., None]
            davis = torch.tensor(frames_np[t], device=dev)
            comp = img * a + davis * (1 - a)
            panel = torch.cat([davis, comp], dim=1)                     # [H, 2W, 3]
            orig_frames.append(to_uint8(panel))
    save_video(orig_frames, os.path.join(args.outdir, "original_view.mp4"), fps=args.fps)
    print(f"  wrote original_view.mp4  ([DAVIS | mesh overlay], {len(orig_frames)} frames)")

    # ---- cyclic orbit (centred, closed loop) ----
    # fixed look-at point = mean over frames of the centred bear centroid (~0); radius fit to size
    with torch.no_grad():
        v0 = world_verts(0, with_root_t=False)
        extent = (v0.max(0).values - v0.min(0).values).max().item()
    radius = 2.4 * max(extent, 1e-3)
    fov = np.deg2rad(40.0)
    Kc = fov_to_intrinsics_normalized(fov, fov, device=dev)
    az0 = 90.0                                                           # start at the side view
    cyc_frames = []
    with torch.no_grad():
        for t in range(T):
            vw = world_verts(t, with_root_t=False)
            center = vw.mean(0)
            vw = vw - center                                             # centre the articulated bear
            frac = t / max(1, T - 1)
            az = np.deg2rad(az0 + args.cyc_amp * np.sin(2 * np.pi * frac))
            el = np.deg2rad(args.cyc_el)
            cam_dir = torch.tensor([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)],
                                   device=dev, dtype=torch.float32)
            eye = radius * cam_dir
            Ec = look_at_extrinsics(eye, [0, 0, 0], up=[0, 0, 1], device=dev)
            img, _ = r.render_color(vw, faces, colors, Ec, Kc, args.cyc_res, args.cyc_res,
                                    ssaa=2, bg=0.93)
            cyc_frames.append(to_uint8(img))
    save_video(cyc_frames, os.path.join(args.outdir, "cyclic_view.mp4"), fps=args.fps)
    print(f"  wrote cyclic_view.mp4  (closed-loop orbit az={az0}+-{args.cyc_amp}deg, {len(cyc_frames)} frames)")

    # contact sheets for quick inspection
    def sheet(frames, k=6, name="sheet.png"):
        idx = np.linspace(0, len(frames) - 1, k).astype(int)
        row = np.concatenate([np.asarray(Image.fromarray(frames[i]).resize((320, int(320 * frames[i].shape[0] / frames[i].shape[1])))) for i in idx], axis=1)
        Image.fromarray(row).save(os.path.join(args.outdir, name))
    sheet(orig_frames, name="original_sheet.png")
    sheet(cyc_frames, name="cyclic_sheet.png")
    print(f"saved renders + contact sheets to {args.outdir}")


if __name__ == "__main__":
    main()
