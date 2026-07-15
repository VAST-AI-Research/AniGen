#!/usr/bin/env python3
"""Text -> camera trajectory. Turn a sentence ("slowly orbit the robot", "dolly in", "hold still")
into a per-frame camera path (c2w[F,4,4] + intrinsics[F,4]) that keeps the edited asset framed, saved
as the .npz (keys c2w, intr) that compose_and_render.py --camera-npy consumes.

A base LLM (Claude, reused from agent_motion) maps the text to a small trajectory spec {type, degrees,
meters, direction}; a deterministic builder turns that into the path (a keyword fallback runs if no
LLM). Aims at the asset's frame-0 world position (from the recon subject mask). Uses the vendored
_media (runs in the AniGen env, CPU).

  python camera_agent.py --recon <recon_dir> --mask <recon>/dynamic_mask --frames 49 \\
      --instruction "slowly orbit the robot ~25 degrees" --out cam.npz
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from agent_motion import llm_call, llm_available, _extract_json   # reuse the LLM client

SYSTEM = """Map a camera-direction request to ONE JSON trajectory spec. The subject stays framed.
Return ONLY: {"type": "static|orbit|dolly_in|dolly_out|pan", "degrees": <float>, "meters": <float>,
"direction": "left|right"}. static = locked; orbit = arc around the subject (degrees, direction);
dolly_in/out = move toward/away (meters); pan = rotate the camera in place (degrees, direction)."""


def _load_recon(recon, mask_dir, F):
    from _media import load_cameras, load_depths, load_masks
    c2w, intr = load_cameras(os.path.join(recon, "cameras.npz")); c2w = c2w[:F].astype(np.float64); intr = intr[:F]
    dep = load_depths(os.path.join(recon, "depths"))[:F].astype(np.float32)
    K0 = np.eye(3); K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2] = intr[0]
    # target = median world point of the subject at frame0 (unproject the masked pixels)
    if mask_dir and os.path.isdir(mask_dir):
        m = load_masks(mask_dir)[0].astype(bool)
    else:
        mf = sorted(glob.glob(os.path.join(mask_dir or "", "*.png")))
        from PIL import Image
        m = (np.asarray(Image.open(mf[0]).convert("L").resize((dep.shape[2], dep.shape[1]))) > 127) if mf else None
    d0 = dep[0]; H, W = d0.shape
    if m is not None and m.shape != (H, W):        # gt-mask PNGs can be any resolution -> match the depth
        from PIL import Image
        m = np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)) > 127
    if m is not None and m.any():
        ys, xs = np.where(m & np.isfinite(d0) & (d0 > 0)); z = d0[ys, xs]
    else:
        ys, xs = np.where(np.isfinite(d0) & (d0 > 0)); z = d0[ys, xs]
    X = (xs - K0[0, 2]) / K0[0, 0] * z; Y = (ys - K0[1, 2]) / K0[1, 1] * z
    world = np.stack([X, Y, z], 1) @ c2w[0][:3, :3].T + c2w[0][:3, 3]
    target = np.median(world, 0)
    up = -c2w[:, :3, 1].mean(0); up = up / (np.linalg.norm(up) + 1e-9)
    return c2w, intr, target, up


def _rot(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-9; x, y, z = a; c, s = np.cos(ang), np.sin(ang)
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])


def build(spec, c2w, intr, target, up, F):
    """Deterministic trajectory builder -> c2w[F,4,4], intr[F,4]. Frame0 == recon frame0 camera."""
    kind = spec.get("type", "static"); deg = float(spec.get("degrees", 25)); m = float(spec.get("meters", 0.6))
    sgn = -1.0 if str(spec.get("direction", "left")).lower().startswith("r") else 1.0
    R0, eye0 = c2w[0][:3, :3].copy(), c2w[0][:3, 3].copy()
    t = np.arange(F) / max(F - 1, 1); ease = t * t * t * (t * (t * 6 - 15) + 10)
    out = np.repeat(c2w[0:1], F, 0).astype(np.float64)
    if kind == "static":
        pass
    elif kind in ("orbit", "pan"):
        for i in range(F):
            Rr = _rot(up, sgn * np.deg2rad(deg) * ease[i])
            if kind == "orbit":                            # rigidly rotate the camera about the subject
                out[i, :3, :3] = Rr @ R0; out[i, :3, 3] = target + Rr @ (eye0 - target)
            else:                                          # pan: rotate in place (eye fixed)
                out[i, :3, :3] = Rr @ R0; out[i, :3, 3] = eye0
    elif kind in ("dolly_in", "dolly_out"):
        fwd = target - eye0; fwd = fwd / (np.linalg.norm(fwd) + 1e-9); d = m * ease * (1 if kind == "dolly_in" else -1)
        for i in range(F):
            out[i, :3, 3] = eye0 + fwd * d[i]
    return out, np.repeat(intr[0:1], F, 0)


def plan(instruction, model="claude-opus-4-8", effort="low"):
    if llm_available():
        try:
            return _extract_json(llm_call(SYSTEM, f"REQUEST: {instruction}\nReturn ONLY the JSON.", model=model, effort=effort))
        except Exception as e:
            print(f"[camera-agent] LLM failed ({e}); keyword fallback", file=sys.stderr)
    s = instruction.lower()                                # keyword fallback
    kind = "orbit" if "orbit" in s or "around" in s else "dolly_in" if "in" in s or "closer" in s else \
           "dolly_out" if "out" in s or "away" in s else "pan" if "pan" in s else "static"
    import re
    deg = float((re.findall(r"(\d+)\s*deg", s) or [25])[0])
    return {"type": kind, "degrees": deg, "meters": 0.6, "direction": "right" if "right" in s else "left"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True); ap.add_argument("--mask", default=None)
    ap.add_argument("--instruction", required=True); ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="claude-opus-4-8")
    a = ap.parse_args()
    spec = plan(a.instruction, a.model)
    print(f"[camera-agent] '{a.instruction}' -> {spec}")
    c2w, intr, target, up = _load_recon(a.recon, a.mask, a.frames)
    C, I = build(spec, c2w, intr, target, up, a.frames)
    np.savez(a.out, c2w=C, intr=I); print(f"[camera-agent] wrote {a.frames}-frame trajectory -> {a.out}")
