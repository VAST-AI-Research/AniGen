"""Prepare a DAVIS-style sequence from extracted frames: foreground masks + first-frame RGBA.

Generates per-frame masks into Annotations/<seq>/ (bear layout) + assets/<seq>_rgba.png for
AniGen.  Backends (all offline): 'birefnet' (BiRefNet-general PyTorch from HF cache — sharp,
excludes background clutter; recommended) or a rembg session name like 'u2net'.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import argparse
import glob

import numpy as np
import torch
from PIL import Image

from paths import davis_paths


def birefnet_soft_masks(imgs, dev="cuda"):
    """Yield (PIL RGB, PIL L soft-mask) per frame using BiRefNet-general (offline HF cache)."""
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    m.eval().to(dev).half()
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for fp in imgs:
        im = Image.open(fp).convert("RGB")
        W, H = im.size
        x = tf(im).unsqueeze(0).to(dev).half()
        with torch.no_grad():
            out = m(x)
            pred = (out[-1] if isinstance(out, (list, tuple)) else out).sigmoid().float().cpu()[0, 0]
        soft = Image.fromarray((pred.numpy() * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        yield im, soft


def rembg_soft_masks(imgs, model):
    from rembg import new_session, remove
    session = new_session(model)
    for fp in imgs:
        im = Image.open(fp).convert("RGB")
        yield im, remove(im, session=session, only_mask=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="birefnet",
                    help="'sam3' (text-prompted video VOS, temporally consistent), 'birefnet' "
                         "(per-frame, offline), or a rembg session name e.g. 'u2net'")
    ap.add_argument("--prompt", default="object", help="text prompt for --model sam3 (e.g. 'robot')")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--asset_frame", type=int, default=0,
                    help="which frame becomes the AniGen input RGBA (-1 = last, for reverse-fit sequences)")
    ap.add_argument("--primary", action="store_true",
                    help="[--model sam3] keep only the PRIMARY tracked object (foreground); drops a 2nd matching object (e.g. a background camel that overlaps the target)")
    args = ap.parse_args()

    frames_dir, ann_dir = davis_paths(args.seq)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(args.assets, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    assert imgs, f"no frames in {frames_dir}"

    if args.model == "sam3":
        from sam3_seg import sam3_video_masks
        frames = np.stack([np.asarray(Image.open(fp).convert("RGB")) for fp in imgs]).astype(np.uint8)
        masks = sam3_video_masks(frames, prompt=args.prompt, primary_only=args.primary)   # [T,H,W] bool
        for i in range(len(imgs)):
            Image.fromarray((masks[i].astype(np.uint8) * 255), "L").save(os.path.join(ann_dir, f"{i:05d}.png"))
        af = args.asset_frame % len(imgs)
        rgba = np.dstack([frames[af], (masks[af].astype(np.uint8) * 255)]).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(args.assets, f"{args.seq}_rgba.png"))
    else:
        gen = birefnet_soft_masks(imgs) if args.model == "birefnet" else rembg_soft_masks(imgs, args.model)
        af = args.asset_frame % len(imgs)
        for i, (im, soft) in enumerate(gen):
            mask = (np.asarray(soft) > 127).astype(np.uint8) * 255
            Image.fromarray(mask, "L").save(os.path.join(ann_dir, f"{i:05d}.png"))
            if i == af:
                rgba = np.dstack([np.asarray(im), np.asarray(soft)]).astype(np.uint8)
                Image.fromarray(rgba, "RGBA").save(os.path.join(args.assets, f"{args.seq}_rgba.png"))
            if i % 20 == 0:
                print(f"  {i+1}/{len(imgs)}  fg frac={float((np.asarray(soft)>127).mean()):.3f}")

    print(f"masks [{args.model}] -> {ann_dir} ({len(imgs)} frames)")
    print(f"asset RGBA (frame {args.asset_frame}) -> {os.path.join(args.assets, args.seq + '_rgba.png')}")


if __name__ == "__main__":
    main()
