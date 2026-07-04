"""Stage 1: render N views of the generated (static) mesh from known cameras.

These renders + the real masked bear image are fed to VGGT-Omega to estimate the object's
viewpoint in the real image.  Cameras use the AniGen canonical Z-up world, OpenCV
world->camera extrinsics, look-at the mesh centroid, up=+Z.  Outputs square RGB on white.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import os

import numpy as np
import torch
from PIL import Image

from geometry import look_at_extrinsics, fov_to_intrinsics_normalized
from renderer import Renderer, to_uint8


def orbit_eyes(center, radius, azimuths_deg, elevations_deg):
    eyes, meta = [], []
    for el in elevations_deg:
        for az in azimuths_deg:
            a = np.deg2rad(az)
            e = np.deg2rad(el)
            d = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)], dtype=np.float64)
            eyes.append(center + radius * d)
            meta.append((az, el))
    return eyes, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="results/bear/rig.npz")
    ap.add_argument("--out", default="results/bear/views")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--radius", type=float, default=2.0)
    args = ap.parse_args()

    dev = "cuda"
    d = np.load(args.rig)
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32)
    colors = torch.tensor(d["vertex_colors"], device=dev, dtype=torch.float32)
    center = verts.mean(0).cpu().numpy().astype(np.float64)

    az = [0, 30, 60, 90, 135, 180, 225, 270, 315]      # around
    el = [10, 45]                                       # two rings
    eyes, meta = orbit_eyes(center, args.radius, az, el)

    fov = np.deg2rad(args.fov)
    K = fov_to_intrinsics_normalized(fov, fov, device=dev)
    r = Renderer(device=dev)

    os.makedirs(args.out, exist_ok=True)
    extr_list, meta_list = [], []
    montage = []
    for i, (eye, (a, e)) in enumerate(zip(eyes, meta)):
        E = look_at_extrinsics(eye, center, up=[0, 0, 1], device=dev)
        with torch.no_grad():
            img, alpha = r.render_color(verts, faces, colors, E, K, args.res, args.res,
                                        near=0.01, far=100.0, ssaa=2, bg=1.0)
        u8 = to_uint8(img)
        Image.fromarray(u8).save(os.path.join(args.out, f"view_{i:02d}.png"))
        extr_list.append(E.cpu().numpy())
        meta_list.append((float(a), float(e)))
        montage.append(u8)
        print(f"  view {i:02d} az={a:>4} el={e:>3}  coverage={alpha.mean().item():.3f}")

    extr = np.stack(extr_list, 0).astype(np.float32)
    np.savez(os.path.join(args.out, "cameras.npz"),
             extrinsics=extr, intrinsics=K.cpu().numpy().astype(np.float32),
             fov_rad=fov, res=args.res, center=center.astype(np.float32),
             radius=args.radius, meta=np.array(meta_list, dtype=np.float32))

    # montage for a quick eyeball check
    cols = len(az)
    rows = len(el)
    m = np.stack(montage, 0).reshape(rows, cols, args.res, args.res, 3)
    m = m.transpose(0, 2, 1, 3, 4).reshape(rows * args.res, cols * args.res, 3)
    Image.fromarray(m).save(os.path.join(args.out, "montage.png"))
    print(f"saved {len(eyes)} views + cameras.npz + montage.png to {args.out}")
    print(f"center={center.round(4).tolist()} radius={args.radius} fov={args.fov}")


if __name__ == "__main__":
    main()
