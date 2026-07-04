"""Make both flanks of the bear share the same (well-observed) texture color.

AniGen only saw one side of the bear in the single input image, so it textures the visible
flank (+y) with correct dark fur but hallucinates the occluded flank (-y) as washed-out
cream.  The bear is bilaterally symmetric about the sagittal plane y=0, so for each vertex we
mirror across y and keep the *darker* (less washed-out) of {self, mirror}.  This copies the
observed fur color onto the unobserved side while leaving genuinely light regions (muzzle,
claws) — whose mirror is also light — untouched.

Writes `vertex_colors_sym` into rig.npz (keeps original as `vertex_colors_raw`) and a
before/after comparison PNG.
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


def mirror_nn(verts):
    """Index of the nearest vertex to each vertex's y-mirror."""
    mirror = verts.copy()
    mirror[:, 1] *= -1.0
    try:
        from scipy.spatial import cKDTree
        return cKDTree(verts).query(mirror, k=1)[1]
    except Exception:
        v = torch.tensor(verts, device="cuda")
        m = torch.tensor(mirror, device="cuda")
        idx = torch.empty(len(m), dtype=torch.long)
        for i in range(0, len(m), 2048):
            idx[i:i + 2048] = torch.cdist(m[i:i + 2048], v).argmin(1).cpu()
        return idx.numpy()


def symmetrize(verts, colors):
    idx = mirror_nn(verts)
    bright = colors.mean(1)
    use_mirror = bright[idx] < bright                       # mirror is darker -> less washed
    out = np.where(use_mirror[:, None], colors[idx], colors).astype(np.float32)
    return out, int(use_mirror.sum())


def render_two(rig_colors, verts, faces, tag, dev="cuda"):
    r = Renderer(dev)
    vt = torch.tensor(verts, device=dev)
    ft = torch.tensor(faces, device=dev, dtype=torch.int32)
    ct = torch.tensor(rig_colors, device=dev)
    center = vt.mean(0)
    K = fov_to_intrinsics_normalized(np.deg2rad(40), np.deg2rad(40), device=dev)
    imgs = []
    for az in (90, 270):
        a, e = np.deg2rad(az), np.deg2rad(10)
        eye = center + 2.0 * torch.tensor([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)],
                                          device=dev, dtype=torch.float32)
        E = look_at_extrinsics(eye, center, up=[0, 0, 1], device=dev)
        with torch.no_grad():
            img, _ = r.render_color(vt, ft, ct, E, K, 512, 512, ssaa=2)
        imgs.append(to_uint8(img))
    return np.concatenate(imgs, axis=1)   # [512, 1024, 3]  (az90 | az270)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="results/bear/rig.npz")
    args = ap.parse_args()

    d = dict(np.load(args.rig))
    verts = d["vertices"].astype(np.float32)
    colors_raw = d["vertex_colors"].astype(np.float32)
    faces = d["faces"]

    colors_sym, n = symmetrize(verts, colors_raw)
    print(f"symmetrized colors: replaced {n}/{len(verts)} vertices with their mirror ({100*n/len(verts):.1f}%)")
    for name, sel in [("+y", verts[:, 1] > 0.02), ("-y", verts[:, 1] < -0.02)]:
        print(f"  {name} flank brightness: raw={colors_raw[sel].mean():.3f} -> sym={colors_sym[sel].mean():.3f}")

    before = render_two(colors_raw, verts, faces, "raw")
    after = render_two(colors_sym, verts, faces, "sym")
    comp = np.concatenate([before, after], axis=0)   # top=raw, bottom=sym; cols = az90 | az270
    Image.fromarray(comp).save(os.path.join(os.path.dirname(args.rig), "symmetrize_compare.png"))

    d["vertex_colors_raw"] = colors_raw
    d["vertex_colors_sym"] = colors_sym
    d["vertex_colors"] = colors_sym                  # default rendering uses symmetrized colors
    np.savez_compressed(args.rig, **d)
    print(f"updated {args.rig} (vertex_colors=symmetrized; raw kept as vertex_colors_raw)")
    print("comparison -> results/bear/symmetrize_compare.png (top=raw, bottom=sym; left=az90 near, right=az270 far)")


if __name__ == "__main__":
    main()
