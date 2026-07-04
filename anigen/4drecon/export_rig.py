"""Stage 1: export the AniGen rig (mesh + skeleton + skin weights) to a single npz.

Runs the AniGen image->3D pipeline once and saves mutually-consistent per-vertex data (vertices
[V,3], faces [F,3], vertex_colors [V,3] in [0,1] sampled from the baked texture, joints [M,3],
parents [M] with -1 for root, skin_weights [V,M] rows summing to 1), all in AniGen's canonical
**Z-up** frame. The textured mesh.glb / skeleton.glb are produced separately by example.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image

sys.path.append(os.getcwd())

from anigen.pipelines import AnigenImageTo3DPipeline
from anigen.utils.random_utils import set_random_seed
from anigen.utils.ckpt_utils import ensure_ckpts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_path', default='assets/bear_rgba.png')
    ap.add_argument('--out', default='results/bear/rig.npz')
    ap.add_argument('--ss_flow_path', default='ckpts/anigen/ss_flow_solo')
    ap.add_argument('--slat_flow_path', default='ckpts/anigen/slat_flow_auto')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cfg_scale_ss', type=float, default=7.5)
    ap.add_argument('--cfg_scale', type=float, default=3.0)
    ap.add_argument('--joints_density', type=int, default=1)
    args = ap.parse_args()

    set_random_seed(args.seed)
    ensure_ckpts()

    print('Loading models...')
    pipe = AnigenImageTo3DPipeline.from_pretrained(
        ss_flow_path=args.ss_flow_path,
        slat_flow_path=args.slat_flow_path,
        device='cuda',
    )
    pipe.cuda()

    img = Image.open(args.image_path)
    print('Running pipeline (texture_size=1024; skin_weights+vertices aligned to UV mesh)...')
    out = pipe.run(
        img,
        seed=args.seed,
        cfg_scale_ss=args.cfg_scale_ss,
        cfg_scale_slat=args.cfg_scale,
        joints_density=args.joints_density,
        texture_size=1024,     # returned vertices/skin_weights are aligned to the UV mesh (line 455)
        output_glb=None,
    )

    mesh = out['mesh']
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Per-vertex colors: sample the baked texture at each vertex's UV (nearest).
    uvs = np.asarray(mesh.visual.uv, dtype=np.float32)          # [V,2]
    tex = np.asarray(out['texture_image'])                       # [Ht,Wt,3] uint8
    th, tw = tex.shape[:2]
    u = np.clip(uvs[:, 0], 0.0, 1.0)
    v = np.clip(uvs[:, 1], 0.0, 1.0)
    px = np.clip(np.round(u * (tw - 1)).astype(np.int64), 0, tw - 1)
    py = np.clip(np.round((1.0 - v) * (th - 1)).astype(np.int64), 0, th - 1)  # texture origin top-left
    colors = tex[py, px, :3].astype(np.float32) / 255.0

    joints = np.asarray(out['joints'], dtype=np.float32)
    parents = np.asarray(out['parents'], dtype=np.int64)
    skin = np.asarray(out['skin_weights'], dtype=np.float32)

    # Sanity
    V, M = verts.shape[0], joints.shape[0]
    assert faces.min() >= 0 and faces.max() < V, 'faces index out of range'
    assert skin.shape == (V, M), f'skin_weights {skin.shape} != ({V},{M})'
    assert colors.shape[0] == V, f'colors {colors.shape} != V={V}'
    # normalize skin rows just in case
    rs = skin.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    skin = skin / rs

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        vertices=verts, faces=faces, vertex_colors=colors, uvs=uvs,
        joints=joints, parents=parents, skin_weights=skin,
    )
    Image.fromarray(tex).save(os.path.join(os.path.dirname(args.out), 'texture.png'))
    out['processed_image'].save(os.path.join(os.path.dirname(args.out), 'processed_image.png'))

    print('==== RIG EXPORT SUMMARY ====')
    print(f'  vertices     : {verts.shape}   bounds min={verts.min(0).round(4).tolist()} max={verts.max(0).round(4).tolist()}')
    print(f'  faces        : {faces.shape}')
    print(f'  vertex_colors: {colors.shape}  range [{colors.min():.3f},{colors.max():.3f}]')
    print(f'  joints       : {joints.shape}  bounds min={joints.min(0).round(4).tolist()} max={joints.max(0).round(4).tolist()}')
    print(f'  parents      : {parents.shape} n_roots={(parents < 0).sum()}  range [{parents.min()},{parents.max()}]')
    print(f'  skin_weights : {skin.shape}  row-sum[min,max]=[{skin.sum(1).min():.3f},{skin.sum(1).max():.3f}]  '
          f'max_per_vertex_joints={(skin > 1e-3).sum(1).max()}')
    print(f'  saved -> {args.out}')


if __name__ == '__main__':
    main()
