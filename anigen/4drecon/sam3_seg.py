"""SAM3 (transformers) text-prompted video object segmentation.

Temporally-consistent masks for a whole clip from a text keyword (open-vocabulary), e.g.
"robot". Segments a downscaled copy for memory, then upsamples masks to the frame resolution.
Requires transformers>=5 with SAM3 + the `facebook/sam3` weights (auto-downloaded once).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports

import numpy as np
import torch
from PIL import Image


def sam3_video_masks(frames_uint8, prompt="object", device="cuda", seg_w=1024, model_id=None, primary_only=False):
    """frames_uint8: [T,H,W,3] uint8 -> masks [T,H,W] bool.

    primary_only=True keeps ONLY the primary tracked object (the largest instance at its first
    appearance) by its SAM3 object-id, so a second matching object (e.g. a background camel that
    overlaps the target) is NOT unioned in -- VOS keeps the target's identity through the overlap.
    Default (False) unions all matched instances.
    """
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model_id = model_id or os.environ.get("SAM3_MODEL_ID", "facebook/sam3")

    T, H, W, _ = frames_uint8.shape
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sh = max(16, int(round(H * seg_w / W)))
    vids = np.stack([np.asarray(Image.fromarray(frames_uint8[i]).resize((seg_w, sh), Image.BILINEAR))
                     for i in range(T)]).astype(np.uint8)

    model = Sam3VideoModel.from_pretrained(model_id).to(dtype=dtype, device=device).eval()
    proc = Sam3VideoProcessor.from_pretrained(model_id)
    session = proc.init_video_session(video=vids, inference_device=device, processing_device=device,
                                      video_storage_device=device, dtype=dtype)
    keywords = [k.strip() for k in (prompt.split(",") if isinstance(prompt, str) else prompt)]
    session = proc.add_text_prompt(inference_session=session, text=keywords)

    small = np.zeros((T, sh, seg_w), dtype=bool)
    per_obj = {}                                                    # fi -> {object_id: mask} (for primary_only)
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=T - 1):
            po = proc.postprocess_outputs(session, out)
            oid = po.get("object_ids", None)
            fi = int(out.frame_idx)
            if oid is not None and getattr(oid, "shape", [0])[0] > 0:
                m = po["masks"].detach().cpu().numpy().astype(bool)   # [n, sh, seg_w]
                if m.ndim == 3 and m.shape[0] > 0:
                    small[fi] = m.any(0)
                    ids = np.asarray(oid).reshape(-1)
                    per_obj[fi] = {int(ids[k]): m[k] for k in range(min(len(ids), m.shape[0]))}
    del model, proc
    torch.cuda.empty_cache()

    if primary_only and per_obj:
        f0 = min(per_obj)                                          # earliest frame with a detection
        prim = max(per_obj[f0], key=lambda o: int(per_obj[f0][o].sum()))   # largest object there = foreground
        small = np.zeros((T, sh, seg_w), dtype=bool)
        for fi, d in per_obj.items():
            if prim in d:
                small[fi] = d[prim]
        print(f"SAM3 primary-only: tracking object id={prim} (foreground); dropped other matched instances")

    masks = np.stack([np.asarray(Image.fromarray((small[i].astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)) > 127
                      for i in range(T)])
    print(f"SAM3 [{keywords}]: mask coverage {100 * masks.mean():.2f}%  "
          f"(empty frames: {int((~masks.reshape(T, -1).any(1)).sum())}/{T})")
    return masks
