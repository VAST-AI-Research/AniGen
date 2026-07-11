"""Stage 8 (optional): a cinematic showcase video for one fitted sequence.

Storyboard (single fixed fitting camera; the object is translated/rotated in world for the
"camera" moves):
  A  hold on the input first frame
  B  fade in the mesh + its boundary contour (frame-0 pose)
  C  fade the background out while the skeleton fades in
  D  translate the mesh+skeleton to screen center
  E  spin 360 deg about the vertical (camera-up) axis
  F  translate back to the original first-frame position
  G  fade the background back in
  H  play the fitted animation (mesh + contour + skeleton over the video)

Output: results/<seq>/renders/showcase.mp4 (+ .gif).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse

import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.cm as cm
from PIL import Image

from geometry import (rot6d_to_matrix, apply_similarity, axis_angle_to_matrix, Skeleton,
                      intrinsics_to_projection)
from renderer import Renderer, to_uint8
from davis import load_davis, davis_paths
from fit_utils import pick_vertex_colors
from render_skeleton import draw_skeleton, draw_mesh_contour, project, CONTOUR_COLORS


def smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def blend(base, layer, f):
    f = float(np.clip(f, 0.0, 1.0))
    if f <= 0:
        return base
    return np.ascontiguousarray((base.astype(np.float32) * (1 - f) + layer.astype(np.float32) * f)
                                .clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bg_dark", type=float, default=0.28)
    ap.add_argument("--contour", default="blue", choices=list(CONTOUR_COLORS))
    ap.add_argument("--ssaa", type=int, default=2)
    # phase durations (frames)
    ap.add_argument("--f_hold", type=int, default=15)
    ap.add_argument("--f_meshin", type=int, default=25)
    ap.add_argument("--f_bgout", type=int, default=30)
    ap.add_argument("--f_center", type=int, default=30)
    ap.add_argument("--f_spin", type=int, default=133)   # 0.75x-slow 360 orbit (more frames at same fps)
    ap.add_argument("--f_back", type=int, default=30)
    ap.add_argument("--f_bgin", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda"
    Rd = f"results/{args.seq}"
    args.rig = args.rig or f"{Rd}/rig.npz"
    args.motion = args.motion or f"{Rd}/motion.npz"
    args.outdir = args.outdir or f"{Rd}/renders"
    frames_dir, ann_dir = davis_paths(args.seq)
    os.makedirs(args.outdir, exist_ok=True)
    ccol = CONTOUR_COLORS[args.contour]

    d = np.load(args.rig)
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32)
    colors_v = torch.tensor(pick_vertex_colors(d), device=dev, dtype=torch.float32)
    weights = torch.tensor(d["skin_weights"], device=dev, dtype=torch.float32)
    parents = np.asarray(d["parents"]).astype(np.int64)
    sk = Skeleton(d["joints"], d["parents"], device=dev)
    M = sk.M
    jcolors = (np.array([cm.turbo(i / max(1, M - 1)) for i in range(M)])[:, :3] * 255).astype(np.uint8)

    m = np.load(args.motion, allow_pickle=True)
    bone6 = torch.tensor(m["bone6"], device=dev, dtype=torch.float32)
    r6 = torch.tensor(m["r6"], device=dev, dtype=torch.float32)
    tg = torch.tensor(m["tg"], device=dev, dtype=torch.float32)
    s = float(m["scale"]); W, H = int(m["W"]), int(m["H"])
    E = torch.tensor(m["E_fit"], device=dev, dtype=torch.float32)
    K = torch.tensor(m["K_norm"], device=dev, dtype=torch.float32)
    T = bone6.shape[0]
    frames_np, _, _, _ = load_davis(frames_dir, ann_dir, H=H, W=W, n_frames=T)

    def posed(t):
        R_local = rot6d_to_matrix(bone6[t])
        v_can = sk.lbs(verts, weights, R_local)
        _, gp = sk.forward_kinematics(R_local)
        Rg = rot6d_to_matrix(r6[t])
        return apply_similarity(v_can, s, Rg, tg[t]), apply_similarity(gp, s, Rg, tg[t])

    v0, j0 = posed(0)
    v0, j0 = v0.detach(), j0.detach()
    c0 = v0.mean(0)                                                # frame-0 world centroid

    # camera-up axis in world (for the vertical 360 spin) and the "move-to-center" translation
    Rc = E[:3, :3]; tcam = E[:3, 3]
    u_world = torch.nn.functional.normalize(-Rc[1, :], dim=0)      # camera "up" (y is down) in world
    fx_px, fy_px = K[0, 0].item() * W, K[1, 1].item() * H
    c0_cam = Rc @ c0 + tcam
    Z0 = c0_cam[2].item()
    p0px, _ = project(c0[None], E, K, W, H)
    dpx, dpy = (W / 2 - p0px[0, 0]), (H / 2 - p0px[0, 1])
    dcam = torch.tensor([dpx * Z0 / fx_px, dpy * Z0 / fy_px, 0.0], device=dev)
    t_center = Rc.T @ dcam                                         # world shift: c0 -> image center

    r = Renderer(dev)
    full = intrinsics_to_projection(K, 0.01, 100.0) @ E

    @torch.no_grad()
    def frame(vw, jw, bg_np, bgf, mesh_fade, skel_fade):
        img, alpha = r.render_color(vw, faces, colors_v, E, K, H, W, ssaa=args.ssaa, bg=1.0)
        a = alpha[..., None] * mesh_fade
        bg = torch.tensor(bg_np, device=dev) * float(bgf)
        comp = to_uint8(bg * (1 - a) + img * a)
        if mesh_fade > 0.02:
            comp = blend(comp, draw_mesh_contour(comp.copy(), alpha.cpu().numpy(), ccol, 2), mesh_fade)
        if skel_fade > 0.02:
            j2d, jz = project(jw, E, K, W, H)
            comp = blend(comp, draw_skeleton(comp.copy(), j2d, parents, jcolors, jz), skel_fade)
        return comp

    f0 = frames_np[0]
    vid = []

    def rigid(theta, tshift):
        Rm = axis_angle_to_matrix(u_world * float(theta))
        vv = (v0 - c0) @ Rm.T + c0 + tshift
        jj = (j0 - c0) @ Rm.T + c0 + tshift
        return vv, jj

    # A: hold input frame
    for _ in range(args.f_hold):
        vid.append(frame(v0, j0, f0, 1.0, 0.0, 0.0))
    # B: fade in mesh + contour
    for i in range(args.f_meshin):
        vid.append(frame(v0, j0, f0, 1.0, smooth(i / (args.f_meshin - 1)), 0.0))
    # C: fade bg out + skeleton in
    for i in range(args.f_bgout):
        e = smooth(i / (args.f_bgout - 1))
        vid.append(frame(v0, j0, f0, 1 - (1 - args.bg_dark) * e, 1.0, e))
    # D: move to center
    for i in range(args.f_center):
        vv, jj = rigid(0.0, smooth(i / (args.f_center - 1)) * t_center)
        vid.append(frame(vv, jj, f0, args.bg_dark, 1.0, 1.0))
    # E: spin 360
    for i in range(args.f_spin):
        vv, jj = rigid(2 * np.pi * (i / args.f_spin), t_center)
        vid.append(frame(vv, jj, f0, args.bg_dark, 1.0, 1.0))
    # F: move back
    for i in range(args.f_back):
        vv, jj = rigid(0.0, (1 - smooth(i / (args.f_back - 1))) * t_center)
        vid.append(frame(vv, jj, f0, args.bg_dark, 1.0, 1.0))
    # G: fade bg in
    for i in range(args.f_bgin):
        e = smooth(i / (args.f_bgin - 1))
        vid.append(frame(v0, j0, f0, args.bg_dark + (1 - args.bg_dark) * e, 1.0, 1.0))
    # H: play the fitted animation over the video at ORIGINAL speed. The mp4 runs at args.fps (30) for a
    # smooth orbit, but the animation content is 15fps (= the fit videos) -> each animation frame is held
    # for args.fps//15 output frames.
    hold = max(1, args.fps // 15)
    for t in range(T):
        vw, jw = posed(t)
        f = frame(vw.detach(), jw.detach(), frames_np[t], 1.0, 1.0, 1.0)
        for _ in range(hold):
            vid.append(f)

    imageio.mimsave(os.path.join(args.outdir, "showcase.mp4"), vid, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimsave(os.path.join(args.outdir, "showcase.gif"),      # keep every frame (full frame rate)
                    [np.asarray(Image.fromarray(f).resize((W // 2, H // 2))) for f in vid],
                    fps=args.fps, loop=0)
    print(f"wrote showcase.mp4/.gif ({len(vid)} frames @ {args.fps}fps) to {args.outdir}")


if __name__ == "__main__":
    main()
