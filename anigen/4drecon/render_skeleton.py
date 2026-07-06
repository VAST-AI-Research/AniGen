"""Stage 6b: visualize the fitted articulated skeleton as a stick figure in the videos.

Per frame we run forward kinematics on the fitted per-bone rotations to get posed joint
positions, apply the global transform, and project them with the fitting camera (original view)
or a cyclic orbit camera (novel view). Bones (parent->child) are drawn as colored lines and
joints as dots, over the DAVIS frame / the rendered mesh.

Outputs (results/<seq>/renders/):
  skeleton_original.mp4/.gif : [DAVIS + skeleton | mesh + skeleton]   (fitting camera)
  skeleton_cyclic.mp4/.gif   : dimmed mesh + skeleton from a closed-loop orbit  (novel views)
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
import matplotlib.cm as cm
from PIL import Image

from geometry import (rot6d_to_matrix, apply_similarity, Skeleton,
                   look_at_extrinsics, fov_to_intrinsics_normalized, intrinsics_to_projection)
from renderer import Renderer, to_uint8
from davis import load_davis, davis_paths


def project(points, E, K, W, H, near=0.01, far=100.0):
    full = intrinsics_to_projection(K, near, far) @ E
    vh = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    clip = vh @ full.T
    ndc = clip[..., :2] / clip[..., 3:4].clamp(min=1e-6)
    px = (0.5 + 0.5 * ndc[..., 0]) * W
    py = (0.5 + 0.5 * ndc[..., 1]) * H
    z = clip[..., 3]                                          # camera depth (w = Z)
    return torch.stack([px, py], -1).cpu().numpy(), z.cpu().numpy()


def draw_skeleton(img, j2d, parents, colors, depth=None, radius=4, lw=3):
    """Draw bones (parent->child) + joint dots on img (uint8 HxWx3, RGB). Returns the array."""
    img = np.ascontiguousarray(img)                          # cv2 needs a contiguous buffer
    order = np.argsort(-depth) if depth is not None else range(len(parents))  # far->near
    for j in order:
        p = int(parents[j])
        c = tuple(int(x) for x in colors[j])
        if p >= 0:
            cv2.line(img, tuple(np.round(j2d[p]).astype(int)), tuple(np.round(j2d[j]).astype(int)),
                     c, lw, cv2.LINE_AA)
    for j in order:
        p = int(parents[j])
        cc = (255, 60, 60) if p < 0 else tuple(int(x) for x in colors[j])
        r = radius + 2 if p < 0 else radius
        cv2.circle(img, tuple(np.round(j2d[j]).astype(int)), r, cc, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(j2d[j]).astype(int)), r, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_mesh_contour(img, mask, color, thickness=2):
    """Outline the mesh silhouette (mask>0.5) on img (uint8 RGB) with a colored contour."""
    img = np.ascontiguousarray(img)
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness, cv2.LINE_AA)
    return img


CONTOUR_COLORS = {"blue": (90, 200, 255), "pink": (255, 120, 200)}  # RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cyc_amp", type=float, default=140.0)
    ap.add_argument("--cyc_el", type=float, default=20.0)
    ap.add_argument("--cyc_res", type=int, default=720)
    ap.add_argument("--bg_dark", type=float, default=0.35, help="background dim factor (mesh+skeleton stay bright)")
    ap.add_argument("--contour", default="blue", choices=list(CONTOUR_COLORS), help="mesh-boundary contour color")
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
    colors_v = torch.tensor(d["vertex_colors"], device=dev, dtype=torch.float32)
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
    r = Renderer(dev)

    def posed(t, root=True):
        R_local = rot6d_to_matrix(bone6[t])
        v_can = sk.lbs(verts, weights, R_local)
        _, gp = sk.forward_kinematics(R_local)                 # posed pivots (canonical)
        t_vec = tg[t] if root else torch.zeros(3, device=dev)
        Rg = rot6d_to_matrix(r6[t])
        return apply_similarity(v_can, s, Rg, t_vec), apply_similarity(gp, s, Rg, t_vec)

    # ---------- original view: skeleton on DAVIS+mesh, and an [original | mesh | skeleton] triptych ----------
    ccol = CONTOUR_COLORS[args.contour]
    orig, sbs = [], []
    with torch.no_grad():
        for t in range(T):
            vw, jw = posed(t, root=True)
            j2d, jz = project(jw, E, K, W, H)
            img, alpha = r.render_color(vw, faces, colors_v, E, K, H, W, ssaa=2, bg=1.0)
            amask = alpha.cpu().numpy()
            davis = torch.tensor(frames_np[t], device=dev)
            a = alpha[..., None]
            mesh_comp = to_uint8(davis * (1 - a) + img * a)
            panelA = (frames_np[t] * 255 * 0.65).astype(np.uint8)   # dim DAVIS for contrast
            panelA = draw_skeleton(panelA, j2d, parents, jcolors, jz, radius=4, lw=3)
            mesh_comp = draw_skeleton(mesh_comp, j2d, parents, jcolors, jz, radius=4, lw=3)
            orig.append(np.concatenate([panelA, mesh_comp], axis=1))
            # triptych: [ original video | mesh overlay (darkened bg + boundary contour) | skeleton overlay ]
            original = (frames_np[t] * 255).astype(np.uint8)
            dark = davis * args.bg_dark
            mesh_only = draw_mesh_contour(to_uint8(dark * (1 - a) + img * a), amask, ccol, thickness=2)
            skel_only = draw_skeleton(to_uint8(dark), j2d, parents, jcolors, jz, radius=4, lw=3)
            sbs.append(np.concatenate([original, mesh_only, skel_only], axis=1))
    imageio.mimsave(os.path.join(args.outdir, "skeleton_original.mp4"), orig, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimsave(os.path.join(args.outdir, "skeleton_original.gif"),
                    [np.asarray(Image.fromarray(f).resize((f.shape[1] // 3, f.shape[0] // 3))) for f in orig],
                    fps=args.fps, loop=0)
    imageio.mimsave(os.path.join(args.outdir, "fit_sidebyside.mp4"), sbs, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimsave(os.path.join(args.outdir, "fit_sidebyside.gif"),
                    [np.asarray(Image.fromarray(f).resize((f.shape[1] // 6, f.shape[0] // 6))) for f in sbs],
                    fps=args.fps, loop=0)
    print(f"  wrote skeleton_original.mp4/.gif + fit_sidebyside.mp4/.gif "
          f"([original | mesh+{args.contour} contour | skeleton], darkened bg)")

    # ---------- cyclic orbit: skeleton + dimmed mesh, novel views ----------
    Kc = fov_to_intrinsics_normalized(np.deg2rad(40), np.deg2rad(40), device=dev)
    res = args.cyc_res
    with torch.no_grad():
        v0, _ = posed(0, root=False)
        extent = (v0.max(0).values - v0.min(0).values).max().item()
    radius = 2.4 * extent
    cyc = []
    with torch.no_grad():
        for t in range(T):
            vw, jw = posed(t, root=False)
            center = vw.mean(0)
            vw = vw - center; jw = jw - center
            frac = t / max(1, T - 1)
            az = np.deg2rad(90 + args.cyc_amp * np.sin(2 * np.pi * frac))
            el = np.deg2rad(args.cyc_el)
            eye = radius * torch.tensor([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)],
                                        device=dev, dtype=torch.float32)
            Ec = look_at_extrinsics(eye, [0, 0, 0], up=[0, 0, 1], device=dev)
            img, alpha = r.render_color(vw, faces, colors_v, Ec, Kc, res, res, ssaa=2, bg=0.95)
            frame = to_uint8(img * 0.45 + 0.55)                # fade mesh so the skeleton pops
            j2d, jz = project(jw, Ec, Kc, res, res)
            frame = draw_skeleton(frame, j2d, parents, jcolors, jz, radius=4, lw=3)
            cyc.append(frame)
    imageio.mimsave(os.path.join(args.outdir, "skeleton_cyclic.mp4"), cyc, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimsave(os.path.join(args.outdir, "skeleton_cyclic.gif"),
                    [np.asarray(Image.fromarray(f).resize((res // 2, res // 2))) for f in cyc], fps=args.fps, loop=0)
    print("  wrote skeleton_cyclic.mp4/.gif (dimmed mesh + skeleton, orbit)")

    for name, vid in [("skeleton_original_sheet.png", orig), ("skeleton_cyclic_sheet.png", cyc)]:
        idx = np.linspace(0, len(vid) - 1, 6).astype(int)
        ph = vid[0].shape[0]; pw = vid[0].shape[1]
        row = np.concatenate([np.asarray(Image.fromarray(vid[i]).resize((int(360 * pw / ph), 360))) for i in idx], 1)
        Image.fromarray(row).save(os.path.join(args.outdir, name))
    print(f"saved skeleton visualizations to {args.outdir}")


if __name__ == "__main__":
    main()
