"""DAVIS-style video loader + RAFT optical flow (torchvision)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import glob

import numpy as np
import torch
from PIL import Image

from paths import davis_paths  # re-exported for convenience


def load_davis(frames_dir, ann_dir, H=540, W=960, n_frames=None):
    """Load DAVIS frames + binary masks resized to (H,W).

    Returns:
        frames  : float32 [N,H,W,3] in [0,1]
        masks   : float32 [N,H,W]   in {0,1}
        names   : list[str]
        orig_hw : (H0,W0)
    """
    fpaths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    apaths = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
    assert len(fpaths) == len(apaths) and len(fpaths) > 0, \
        f"frame/mask count mismatch {len(fpaths)} vs {len(apaths)}"
    if n_frames is not None:
        fpaths, apaths = fpaths[:n_frames], apaths[:n_frames]

    orig = Image.open(fpaths[0])
    orig_hw = (orig.height, orig.width)

    frames, masks, names = [], [], []
    for fp, ap in zip(fpaths, apaths):
        im = Image.open(fp).convert("RGB").resize((W, H), Image.BILINEAR)
        frames.append(np.asarray(im, dtype=np.float32) / 255.0)
        an = Image.open(ap).resize((W, H), Image.NEAREST)
        am = np.asarray(an)
        if am.ndim == 3:
            am = am[..., 0]
        masks.append((am > 0).astype(np.float32))
        names.append(os.path.splitext(os.path.basename(fp))[0])

    frames = np.stack(frames, 0)
    masks = np.stack(masks, 0)
    return frames, masks, names, orig_hw


def mask_stats(mask: np.ndarray):
    """centroid (cx,cy) in pixels, area fraction, bbox (x0,y0,x1,y1). mask HxW in {0,1}."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        H, W = mask.shape
        return (W / 2.0, H / 2.0), 0.0, (0, 0, W, H)
    cx, cy = xs.mean(), ys.mean()
    area = float(len(xs)) / mask.size
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (float(cx), float(cy)), area, bbox


# --------------------------------------------------------------------------- #
# RAFT optical flow (torchvision)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_raft_flow(frames_np: np.ndarray, device="cuda", pad_to=8):
    """Dense forward flow t->t+1 for a frame stack.

    frames_np : [N,H,W,3] float [0,1]
    Returns   : flow [N-1,H,W,2] float32 (pixel dx,dy, top-left convention), on CPU.
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()

    N, H, W, _ = frames_np.shape
    # RAFT expects sizes divisible by 8.
    Hp = (H + pad_to - 1) // pad_to * pad_to
    Wp = (W + pad_to - 1) // pad_to * pad_to

    def prep(a):
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)      # [1,3,H,W] in [0,1]
        t = torch.nn.functional.interpolate(t, size=(Hp, Wp), mode="bilinear", align_corners=False)
        return t * 2.0 - 1.0                                           # RAFT wants [-1,1]

    flows = []
    for t in range(N - 1):
        img1 = prep(frames_np[t])
        img2 = prep(frames_np[t + 1])
        flow = model(img1, img2)[-1]                                   # [1,2,Hp,Wp]
        # resize flow back to (H,W) and rescale magnitudes
        flow = torch.nn.functional.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow[:, 0] *= W / Wp
        flow[:, 1] *= H / Hp
        flows.append(flow[0].permute(1, 2, 0).cpu())                   # [H,W,2]
    del model
    torch.cuda.empty_cache()
    return torch.stack(flows, 0).numpy().astype(np.float32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    a = ap.parse_args()
    fd, ad = davis_paths(a.seq)
    fr, mk, nm, hw = load_davis(fd, ad, H=270, W=480, n_frames=4)
    print("frames", fr.shape, "masks", mk.shape, "orig", hw, "names", nm)
    for i in range(len(mk)):
        print(" ", nm[i], "maskstats", mask_stats(mk[i]))
    fl = compute_raft_flow(fr, device="cuda")
    print("flow", fl.shape, "mean|flow|", np.abs(fl).mean())
