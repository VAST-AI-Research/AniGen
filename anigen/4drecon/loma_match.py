"""Thin wrapper around the LoMa local feature matcher (https://github.com/davnords/LoMa).

LoMa (DaD detector + DeDoDe-G descriptor + transformer matcher) gives appearance-based
correspondences between the rendered textured mesh and the masked GT frame. Set ``LOMA_ROOT`` to the
checkout (default: ``extensions/LoMa``); the matcher weights live at ``$LOMA_ROOT/loma_G.pth`` and the
aux weights are fetched once via ``torch.hub``.
"""
import os
import sys
import numpy as np
import torch


def _loma_root():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    cand = [os.environ.get("LOMA_ROOT", ""),
            os.path.join(repo_root, "extensions", "LoMa"),
            os.path.abspath(os.path.join(repo_root, "..", "LoMa"))]
    for c in cand:
        if c and os.path.isdir(os.path.join(c, "src", "loma")):
            return c
    raise FileNotFoundError("LoMa checkout not found; set LOMA_ROOT (default extensions/LoMa)")


def load_loma(device="cuda", variant="G"):
    """Construct LoMa and load the local matcher weights.  Returns the eval()'d model."""
    root = _loma_root()
    if os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))
    from loma.loma import LoMa, LoMaG, LoMaB

    cfgs = {"G": LoMaG, "B": LoMaB}
    Cfg = cfgs[variant]
    # weights_url=None -> skip the 1.5 GB hub download; load the local checkpoint instead.
    cfg = Cfg(weights_url=None)
    model = LoMa(cfg).to(device).eval()
    local = os.path.join(root, f"loma_{variant}.pth")
    if os.path.isfile(local):
        sd = torch.load(local, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # 'transformers.N/log_assignment.N' beyond n_layers are allowed extra layers
        bad = [k for k in unexpected if not k.startswith(("transformers.", "log_assignment."))]
        assert not missing and not bad, f"loma load: missing={missing[:4]} unexpected={bad[:4]}"
    else:                                                       # fall back to hub download
        model = LoMa(Cfg()).to(device).eval()
    return model


def _to_input(img, device, mult=14, long_side=1024):
    """np HxWx3 (uint8 or float[0,1]) -> (tensor[1,3,H2,W2] in [0,1], (H2,W2), (scale_x,scale_y))."""
    import cv2
    a = img.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    H, W = a.shape[:2]
    sc = long_side / max(H, W)
    H2 = max(mult, int(round(H * sc / mult)) * mult)
    W2 = max(mult, int(round(W * sc / mult)) * mult)
    a2 = cv2.resize(a, (W2, H2), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(a2).permute(2, 0, 1).float().to(device)[None].clamp(0, 1)
    return t, (H2, W2), (W / W2, H / H2)


@torch.inference_mode()
def match(model, imgA, imgB, device="cuda", num_keypoints=2048, filter_threshold=None):
    """Match imgA vs imgB (np HxWx3).  Returns (ptsA, ptsB) in *original imgA/imgB pixel coords*."""
    tA, _, (sxA, syA) = _to_input(imgA, device)
    tB, _, (sxB, syB) = _to_input(imgB, device)
    pA, pB = model.match(tA, tB, filter_threshold=filter_threshold, num_keypoints=num_keypoints)
    pA = pA.astype(np.float64) * [sxA, syA]
    pB = pB.astype(np.float64) * [sxB, syB]
    return pA, pB


@torch.inference_mode()
def match_scores(model, imgA, imgB, device="cuda", num_keypoints=2048, filter_threshold=None):
    """Like match() but also returns the LoMa confidence score of each matched pair -> (pA, pB, scores)."""
    from loma.loma import filter_matches, to_pixel_coords
    tA, (h1, w1), (sxA, syA) = _to_input(imgA, device)
    tB, (h2, w2), (sxB, syB) = _to_input(imgB, device)
    kA, dA, h1, w1 = model.detect_and_describe(tA, num_keypoints)
    kB, dB, h2, w2 = model.detect_and_describe(tB, num_keypoints)
    if filter_threshold is None:
        filter_threshold = model.cfg.filter_threshold
    scores = model(kA, kB, dA, dB)["scores"]
    m0, _, mscores0, _ = filter_matches(scores, filter_threshold)
    valid = m0[0] > -1
    mA = kA[0][torch.where(valid)[0]]; mB = kB[0][m0[0][valid]]
    sc = mscores0[0][valid].detach().float().cpu().numpy()
    pA = to_pixel_coords(mA, h1, w1).cpu().numpy().astype(np.float64) * [sxA, syA]
    pB = to_pixel_coords(mB, h2, w2).cpu().numpy().astype(np.float64) * [sxB, syB]
    return pA, pB, sc
