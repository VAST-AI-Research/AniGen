"""Run CoTracker3 (offline) on a DAVIS sequence -> results/<seq>/cotracker.npz + official viz.

CoTracker3 is a state-of-the-art *2D* point tracker (no depth). Query points are a grid on the
frame-0 object mask. Requires a CoTracker checkout + checkpoint (see README); paths from paths.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import glob

import numpy as np
import torch
from PIL import Image

from paths import davis_paths, COTRACKER_REPO, COTRACKER_CKPT

CKPT = COTRACKER_CKPT


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n_frames", type=int, default=0, help="0 = all")
    ap.add_argument("--grid_size", type=int, default=40)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=540)
    args = ap.parse_args()
    dev = "cuda"
    out = args.out or f"results/{args.seq}/cotracker.npz"
    outdir = args.outdir or f"results/{args.seq}/renders"
    img_dir, ann_dir = davis_paths(args.seq)
    sys.path.insert(0, COTRACKER_REPO)
    from cotracker.predictor import CoTrackerPredictor
    from cotracker.utils.visualizer import Visualizer

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    anns = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
    if args.n_frames and args.n_frames > 0:
        imgs = imgs[:args.n_frames]; anns = anns[:args.n_frames]
    n = len(imgs)
    frames = np.stack([np.asarray(Image.open(imgs[i]).convert("RGB").resize((args.W, args.H), Image.BILINEAR))
                       for i in range(n)]).astype(np.uint8)                 # [T,H,W,3]
    mask0 = (np.asarray(Image.open(anns[0]).resize((args.W, args.H), Image.NEAREST)) > 0).astype(np.float32)
    if mask0.ndim == 3:
        mask0 = mask0[..., 0]

    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(dev)   # [1,T,C,H,W] 0-255
    segm = torch.from_numpy(mask0)[None, None].to(dev)                            # [1,1,H,W]

    model = CoTrackerPredictor(checkpoint=CKPT, offline=True).to(dev)
    pred_tracks, pred_vis = model(video, grid_size=args.grid_size, segm_mask=segm)
    # drop the support grid (appended when segm_mask is given): keep only on-mask queries
    t0 = pred_tracks[0, 0]                                                        # [N,2] frame-0
    col = t0[:, 0].round().long().clamp(0, args.W - 1)
    row = t0[:, 1].round().long().clamp(0, args.H - 1)
    on = segm[0, 0][row, col] > 0.5
    tracks = pred_tracks[0][:, on].cpu().numpy()                                  # [T,N,2]
    vis = pred_vis[0][:, on].float().cpu().numpy()                               # [T,N]
    print(f"CoTracker3 [{args.seq}]: {tracks.shape[1]} tracks (of {pred_tracks.shape[2]}), "
          f"vis>0.5 frac={float((vis > 0.5).mean()):.2f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, tracks=tracks.astype(np.float32), vis=vis.astype(np.float32),
                        hw=np.array([args.H, args.W], np.int64), frame_idx=np.arange(n, dtype=np.int64))
    print(f"saved -> {out}")

    # official CoTracker visualization
    os.makedirs(outdir, exist_ok=True)
    viser = Visualizer(save_dir=outdir, pad_value=0, linewidth=2, tracks_leave_trace=8)
    on_t = torch.from_numpy(np.where(on.cpu().numpy())[0])
    viser.visualize(video=video.cpu(), tracks=pred_tracks[:, :, on].cpu(),
                    visibility=pred_vis[:, :, on].cpu(), filename="cotracker_official")
    print(f"official overlay -> {outdir}/cotracker_official.mp4")


if __name__ == "__main__":
    main()
