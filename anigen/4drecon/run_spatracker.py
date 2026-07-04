"""(Optional) Run SpatialTracker V2 on a DAVIS sequence -> results/<seq>/spatrack.npz.

Self-contained: drives the SpatialTracker V2 checkout directly (VGGT4Track front-end + Offline
tracker), no external wrapper.  Produces 3D world tracks + visibility/confidence + per-frame
camera, aligned 1:1 to the DAVIS frames.  CoTracker3 (run_cotracker.py) is the recommended default;
this is kept for the optional 3D-track variant.  Requires the SpaTrackerV2 checkout + checkpoints
(see README) + pycolmap/pyceres; all paths from paths.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import glob

import numpy as np
import torch
from PIL import Image

from paths import davis_paths, SPATRACKER_REPO, SPATRACKER_FRONT_CKPT, SPATRACKER_OFFLINE_CKPT


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_frames", type=int, default=0, help="0 = all")
    ap.add_argument("--grid_size", type=int, default=40)
    ap.add_argument("--vo_points", type=int, default=2048)
    ap.add_argument("--load_w", type=int, default=1280, help="width to load frames at (>=518)")
    args = ap.parse_args()
    dev = "cuda"
    out = args.out or os.path.join("results", args.seq, "spatrack.npz")
    img_dir, ann_dir = davis_paths(args.seq)

    sys.path.insert(0, SPATRACKER_REPO)
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.utils import get_points_on_a_grid
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    anns = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
    ntot = min(len(imgs), len(anns))
    n = ntot if args.n_frames <= 0 or args.n_frames > ntot else args.n_frames
    idx = list(range(n)) if n == ntot else sorted(set(np.linspace(0, ntot - 1, n).round().astype(int).tolist()))
    W0, H0 = Image.open(imgs[idx[0]]).size
    lw = args.load_w; lh = int(round(H0 * lw / W0))
    frames = np.stack([np.asarray(Image.open(imgs[i]).convert("RGB").resize((lw, lh), Image.BILINEAR))
                       for i in idx]).astype(np.uint8)
    mask0 = np.asarray(Image.open(anns[idx[0]]).resize((lw, lh), Image.NEAREST)) > 0
    if mask0.ndim == 3:
        mask0 = mask0[..., 0]

    # --- front-end: depth + poses + intrinsics ---
    front = VGGT4Track.from_pretrained(SPATRACKER_FRONT_CKPT).eval().to(dev)
    video = preprocess_image(torch.from_numpy(frames).permute(0, 3, 1, 2).float())[None]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = front(video.to(dev) / 255.0)
    extrs = pred["poses_pred"].squeeze(0).float().cpu().numpy()
    intrs = pred["intrs"].squeeze(0).float().cpu().numpy()
    depth = pred["points_map"][..., 2].squeeze(0).float().cpu().numpy()
    unc = (pred["unc_metric"].squeeze(0).float().cpu().numpy() > 0.5)
    del front; torch.cuda.empty_cache()
    video_t = video.squeeze(0)
    T, _, H, W = video_t.shape

    grid = get_points_on_a_grid(args.grid_size, (H, W), device="cpu")[0]
    m = np.asarray(Image.fromarray(mask0.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    gi = grid.round().long(); gi[:, 0].clamp_(0, W - 1); gi[:, 1].clamp_(0, H - 1)
    grid = grid[torch.from_numpy(m[gi[:, 1].numpy(), gi[:, 0].numpy()])]
    query_xyt = torch.cat([torch.zeros(grid.shape[0], 1), grid], dim=1).numpy()

    # --- tracker ---
    model = Predictor.from_pretrained(SPATRACKER_OFFLINE_CKPT)
    model.spatrack.track_num = args.vo_points
    model.eval().to(dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        (c2w, _intr, _pmap, _confd, track3d, _track2d, vis, conf, _vid) = model.forward(
            video_t, depth=depth, intrs=intrs, extrs=extrs, queries=query_xyt,
            fps=1, full_point=False, iters_track=4, query_no_BA=True, fixed_cam=False,
            stage=1, unc_metric=unc, support_frame=T - 1, replace_ratio=0.2)
    c2w = c2w.float().cpu()
    tr_cam = track3d[:, :, :3].float().cpu()
    world = (torch.einsum("tij,tnj->tni", c2w[:, :3, :3], tr_cam) + c2w[:, :3, 3][:, None]).numpy()
    intrs_out = _intr.float().cpu().numpy() if torch.is_tensor(_intr) else np.asarray(_intr)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        world_tracks=world.astype(np.float32),
        vis=vis.float().cpu().numpy()[..., 0], conf=conf.float().cpu().numpy()[..., 0],
        c2w=c2w.numpy().astype(np.float32), K=intrs_out.astype(np.float32),
        track_rgb=track3d[:, :, 3:6].float().cpu().numpy(),
        hw=np.array([H, W], np.int64), frame_idx=np.array(idx, np.int64),
    )
    print(f"saved -> {out}  (N={world.shape[1]} tracks, {T} frames, vis>0.5 "
          f"frac={float((vis.float().cpu().numpy()[..., 0] > 0.5).mean()):.2f})")


if __name__ == "__main__":
    main()
