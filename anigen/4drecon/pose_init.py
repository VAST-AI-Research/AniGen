"""Stage 3: estimate the object's viewpoint in the real image with VGGT-Omega.

Feeds N mesh renders (known cameras) + the real masked object (white bg) jointly to VGGT-Omega,
which solves them as one rigid multi-view scene; we recover the world-gauge rotation
G (canonical -> VGGT) by rotation-averaging over the renders, then map VGGT's real-image rotation
into the canonical frame:  R_real_canon = S_real @ G.  Requires a VGGT-Omega checkout + checkpoint
(see README; paths from paths.py).

Outputs results/<seq>/vggt_pose.npz (R_real_canon = object->camera rotation) + a residual metric
and a sanity render from the estimated viewpoint.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse

import numpy as np
import torch
from PIL import Image

from paths import VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT
from geometry import (project_to_so3, look_at_extrinsics, fov_to_intrinsics_normalized)
from renderer import Renderer, to_uint8

VGGT_CKPT = VGGT_OMEGA_CKPT


def geodesic_deg(Ra, Rb):
    R = Ra.transpose(-1, -2) @ Rb
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1, 3)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def make_real_input(rgba_path, out_path, res=512, pad=1.15):
    """Composite the RGBA bear on white, square-crop around alpha bbox, resize -> match renders."""
    im = Image.open(rgba_path).convert("RGBA")
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a > 30)
    cx, cy = xs.mean(), ys.mean()
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.5 * pad
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    rgb = np.asarray(im)[..., :3].astype(np.float32)
    alpha = (a.astype(np.float32) / 255.0)[..., None]
    comp = rgb * alpha + 255.0 * (1 - alpha)
    H, W = comp.shape[:2]
    canvas = np.full((y1 - y0, x1 - x0, 3), 255.0, np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = comp[sy0:sy1, sx0:sx1]
    out = Image.fromarray(canvas.clip(0, 255).astype(np.uint8)).resize((res, res), Image.BILINEAR)
    out.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="results/bear/rig.npz")
    ap.add_argument("--views", default="results/bear/views")
    ap.add_argument("--real_rgba", default="assets/bear_rgba.png")
    ap.add_argument("--out", default="results/bear/vggt_pose.npz")
    ap.add_argument("--view_indices", default="0,1,2,3,4,5,6,7,8",
                    help="which render views to feed VGGT (default: the el=10 ring)")
    args = ap.parse_args()
    dev = "cuda"

    cams = np.load(os.path.join(args.views, "cameras.npz"))
    idxs = [int(x) for x in args.view_indices.split(",")]
    render_paths = [os.path.join(args.views, f"view_{i:02d}.png") for i in idxs]
    R_known = torch.tensor(cams["extrinsics"][idxs, :3, :3], device=dev, dtype=torch.float32)  # [N,3,3] w2c

    real_path = os.path.join(os.path.dirname(args.out), "real_input.png")
    make_real_input(args.real_rgba, real_path)
    image_paths = render_paths + [real_path]
    print(f"VGGT input: {len(render_paths)} renders + 1 real = {len(image_paths)} frames")

    sys.path.insert(0, VGGT_OMEGA_REPO)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = load_and_preprocess_images(image_paths, mode="balanced", image_resolution=512, patch_size=16)
    print("preprocessed images:", tuple(images.shape))

    model = VGGTOmega().eval().requires_grad_(False)
    sd = torch.load(VGGT_CKPT, map_location="cpu")
    model.load_state_dict(sd)
    del sd
    model.to(dev)

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(images.to(dev))
    pose_enc = pred["pose_enc"].float()                              # [1,S,9]
    extr_w2c, intr = encoding_to_camera(pose_enc, images.shape[-2:])  # [1,S,3,4],[1,S,3,3]
    S = extr_w2c[0, :, :3, :3].float().to(dev)                        # [S,3,3] VGGT w2c rotations
    del model, pred
    torch.cuda.empty_cache()

    N = len(render_paths)
    S_render = S[:N]                                                 # [N,3,3]
    S_real = S[N]                                                    # [3,3]

    # Gauge rotation G (canonical -> VGGT): G = mean_i S_i^T R_i
    Gs = torch.einsum("nij,njk->nik", S_render.transpose(-1, -2), R_known)  # [N,3,3]
    G = project_to_so3(Gs.mean(0))

    # residual: how well VGGT reproduces each known render rotation via S_i @ G
    R_pred = torch.einsum("nij,jk->nik", S_render, G)
    res = geodesic_deg(R_pred, R_known)
    print(f"gauge residual over renders: mean={res.mean():.2f} deg  max={res.max():.2f} deg")

    R_real_canon = (S_real @ G)                                     # object->camera rotation, canonical world

    # interpret viewpoint (camera direction from origin)
    fwd_world = R_real_canon.T @ torch.tensor([0., 0., 1.], device=dev)  # OpenCV +Z forward
    cam_dir = -fwd_world                                            # origin -> camera
    az = float(np.rad2deg(np.arctan2(cam_dir[1].item(), cam_dir[0].item())))
    el = float(np.rad2deg(np.arcsin(cam_dir[2].clamp(-1, 1).item())))
    print(f"estimated viewpoint: az={az:.1f} deg  el={el:.1f} deg")

    # sanity render from the estimated viewpoint
    d = np.load(args.rig)
    verts = torch.tensor(d["vertices"], device=dev)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32)
    colors = torch.tensor(d["vertex_colors"], device=dev)
    center = verts.mean(0)
    r = 2.0
    eye = center + r * cam_dir
    E_sanity = look_at_extrinsics(eye, center, up=[0, 0, 1], device=dev)
    K = fov_to_intrinsics_normalized(np.deg2rad(40), np.deg2rad(40), device=dev)
    rr = Renderer(dev)
    with torch.no_grad():
        img, _ = rr.render_color(verts, faces, colors, E_sanity, K, 512, 512, ssaa=2)
    # side-by-side with the real input
    real_im = np.asarray(Image.open(real_path).resize((512, 512)))
    combo = np.concatenate([to_uint8(img), real_im], axis=1)
    Image.fromarray(combo).save(os.path.join(os.path.dirname(args.out), "vggt_sanity.png"))

    np.savez(args.out,
             R_real_canon=R_real_canon.cpu().numpy().astype(np.float32),
             G=G.cpu().numpy().astype(np.float32),
             cam_dir=cam_dir.cpu().numpy().astype(np.float32),
             az=az, el=el, residual_deg=res.mean().item(),
             view_indices=np.array(idxs))
    print(f"saved -> {args.out}  (sanity: vggt_sanity.png = [estimated | real])")


if __name__ == "__main__":
    main()
