#!/usr/bin/env python3
"""STEP 1 (option B) - author a NEW motion from a TEXT instruction with a base LLM (Claude Opus / etc).

This is the "agent" edit path: instead of hand-coding a motion (see reanimate.py) or key-framing it in
Blender, you type what you want ("do jumping jacks", "wave the right arm", "rear up like a horse") and a
base model looks at the asset's SKELETON (joint positions + hierarchy + L/R/region labels) and returns
a small JSON "motion program". A deterministic executor turns that program
into per-joint delta rotations about canonical axes -> bone6_new -> a colored point-cloud sequence,
reusing exactly the primitives in reanimate.py. Frame-0 always equals the original (every curve is 0 at
t=0), which is required by the downstream compositing step.

The LLM is Claude via the Anthropic API (set ANTHROPIC_API_KEY). Any base model works if you swap
llm_call(). Runs on CPU. Typical use:

  export ANTHROPIC_API_KEY=sk-ant-...
  python agent_motion.py unitree_g1_1 "do 2 jumping jacks with a small hop" \\
      --frames 49 --export /tmp/robot_jj_pc --full-texture \\
      --preview /tmp/robot_jj.gif --save-motion results/unitree_g1_1/reanim_jj.npz

  # inspect / iterate on the program without calling the model again:
  python agent_motion.py unitree_g1_1 "..." --dry-run                 # print the program, don't render
  python agent_motion.py unitree_g1_1 "..." --program prog.json ...   # execute a saved/edited program
"""
import os, sys, json, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from reanimate import (load_asset, frame0_globals, build_delta_global, compose,
                       pose_sequence, export_pc, preview_gif, ramp_hold_return)

# --------------------------------------------------------------------------- LLM client (Anthropic API)
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")   # any Claude model id works


def llm_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_call(system, user_text, model=DEFAULT_MODEL, effort="high", max_tokens=8000, timeout=300):
    """One Claude call via the Anthropic API. Set ANTHROPIC_API_KEY (and optionally ANTHROPIC_BASE_URL to
    point at a compatible endpoint). Returns the text response. `effort` is accepted for call-site
    compatibility but not sent to the public API."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("set ANTHROPIC_API_KEY to use the text->motion agent, or pass --program a hand-written JSON")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    payload = {"model": model, "max_tokens": max(max_tokens, 4096), "system": system,
               "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}]}
    req = urllib.request.Request(base + "/v1/messages", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json", "x-api-key": key,
                                          "anthropic-version": "2023-06-01"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")


def _extract_json(text):
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s:
        return json.loads(text[s:e + 1])
    raise ValueError("no JSON object found in LLM response:\n" + text[:500])


# --------------------------------------------------------------------------- skeleton -> LLM context
def describe_skeleton(rig):
    """Human/LLM-readable skeleton table in the CANONICAL frame. For each joint: position, parent, the
    unit direction toward its children (which way the bone points), a left/right/center side label and a
    coarse body-region guess, plus the mirror-symmetric partner joint. This is what lets the model map a
    verbal body part ('right shoulder', 'left hip') onto a joint index."""
    J = np.asarray(rig["joints"], np.float64); P = np.asarray(rig["parents"]); M = J.shape[0]
    children = {j: [k for k in range(M) if P[k] == j] for j in range(M)}
    def _subtree(j):                                      # descendant count incl self
        return 1 + sum(_subtree(k) for k in children[j])
    z = J[:, 2]; zlo, zhi = np.percentile(z, 20), np.percentile(z, 80)
    # symmetric partner = nearest joint to the x-mirrored position
    mirror = []
    for j in range(M):
        q = J.copy(); d = np.linalg.norm(J - np.array([-J[j, 0], J[j, 1], J[j, 2]]), axis=1); d[j] = 1e9
        mirror.append(int(np.argmin(d)) if J[j, 0] != 0 else -1)
    rows = ["idx | pos(x,y,z)          | parent | child_dir(x,y,z)    | side | region  | subtree | mirror"]
    for j in range(M):
        cd = np.zeros(3)
        if children[j]:
            cd = (J[children[j]].mean(0) - J[j]); n = np.linalg.norm(cd); cd = cd / n if n > 1e-6 else cd
        side = "L" if J[j, 0] > 0.03 else ("R" if J[j, 0] < -0.03 else "C")
        region = "upper" if z[j] >= zhi else ("lower" if z[j] <= zlo else "mid")
        rows.append(f"{j:3d} | ({J[j,0]:+.2f},{J[j,1]:+.2f},{J[j,2]:+.2f}) | {int(P[j]):6d} | "
                    f"({cd[0]:+.2f},{cd[1]:+.2f},{cd[2]:+.2f}) | {side:^4} | {region:^7} | {_subtree(j):7d} | {mirror[j]:3d}")
    return "\n".join(rows)


SYSTEM = """You are a character animator. You are given the SKELETON of a rigged 3D asset in its
CANONICAL frame and must translate a natural-language motion request into a JSON "motion program".

Canonical frame (right-handed): +Z = up, -Z = down (toward feet/ground); the X axis is the lateral
(left-right) axis; the Y axis is the fore/aft axis. "side" column: L = +X side, R = -X side, C = center.
"child_dir" is the unit vector from the joint toward its child joints (which way the limb segment points
from that joint) - use it to reason about where a limb will swing.

You author the motion as per-joint DELTA rotations applied on top of the fitted pose. Each joint edit
rotates that joint AND its whole subtree about a chosen GLOBAL canonical axis, by the right-hand rule
(positive angle = counter-clockwise looking down the +axis). Choose the axis and the SIGN so the limb
moves the way the request implies; use the child_dir to get the sign right, and use OPPOSITE signs for
the left vs right joint of a bilateral (symmetric) motion. Rotate the joint that is the ROOT of the
subtree you want to move (e.g. a shoulder to move the whole arm, a hip to move the whole leg). To move
a limb, pick the joint OUT ALONG the limb whose child_dir points away from the torso down the limb (the
shoulder / hip), NOT the small clavicle/pelvis stub next to the spine near center (side=C) - rotating
the stub pivots the limb about the body center and looks lopsided. For a symmetric both-sides motion
pick a true mirror PAIR (see the 'mirror' column) so the two sides match; 'subtree' is the descendant
count (a whole arm/leg is a mid-size subtree, a hand/foot tip is tiny).

Every motion must return to / cycle through the start so that frame 0 equals the original pose. Curves:
- "oscillate": (1-cos(2*pi*reps*t))/2, starts and ends at 0, peaks at amplitude; use for rhythmic
  open-close motions (jumping jacks, waving, bouncing). Set "reps" = number of cycles.
- "sine": sin(2*pi*reps*t), oscillates +/- around 0; use for paw/swing about a neutral pose.
- "ramp_hold_return": eases 0 -> amplitude, holds, eases back to 0; use for a pose-and-hold (rear up,
  reach up, bow). Optional "up" and "hold" are fractions of the clip (defaults 0.3, 0.3).

Return ONLY a JSON object, no prose:
{
  "description": "<short label>",
  "joint_edits": [
    {"joint": <int>, "axis": "x"|"y"|"z", "amplitude_deg": <float, signed>,
     "curve": "oscillate"|"sine"|"ramp_hold_return", "reps": <float, optional>,
     "up": <float optional>, "hold": <float optional>, "comment": "<which body part / intent>"}
  ],
  "root_translation": {"axis": "z", "amplitude_m": <float>, "curve": "oscillate", "reps": <float>}  // optional, e.g. a hop
}
Pick amplitudes in degrees (e.g. an arm raised fully overhead from the side is ~140-160 deg). Keep it
physically plausible. Include every joint you need (both sides for symmetric motion)."""

AXES = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}


def _axis_vec(a):
    if isinstance(a, str):
        return AXES[a.lower()]
    v = np.asarray(a, np.float64); return v / (np.linalg.norm(v) + 1e-9)


def build_curve(name, T, amp, reps=1.0, up=0.3, hold=0.3):
    """Return a length-T curve scaled by amp; guaranteed curve[0]=0 so frame0 == original."""
    t = np.arange(T) / max(T - 1, 1)
    if name == "oscillate":
        c = (1 - np.cos(2 * np.pi * reps * t)) / 2
    elif name == "sine":
        c = np.sin(2 * np.pi * reps * t)
    elif name == "ramp_hold_return":
        c = ramp_hold_return(T, up=up, hold=hold)
    elif name == "linear":
        c = t
    else:
        raise ValueError(f"unknown curve '{name}'")
    return amp * c


def program_to_motion(prog, rig, mot, T):
    """Execute a motion program -> (bone6_new[T,M,6], r6[T,6], tg[T,3]). Root frozen at frame0 + optional hop."""
    M = rig["joints"].shape[0]
    GR0 = frame0_globals(rig, mot["bone6"][0])
    edits = []
    for e in prog.get("joint_edits", []):
        amp = np.deg2rad(float(e["amplitude_deg"]))
        curve = build_curve(e.get("curve", "oscillate"), T, amp, float(e.get("reps", 1.0)),
                            float(e.get("up", 0.3)), float(e.get("hold", 0.3)))
        edits.append((int(e["joint"]), _axis_vec(e["axis"]), curve))
    R_delta = build_delta_global(T, M, rig["parents"], GR0, edits)
    bone6_new = compose(mot["bone6"][0], R_delta)
    r6 = np.repeat(mot["r6"][0:1], T, axis=0)
    tg = np.repeat(mot["tg"][0:1], T, axis=0).copy()
    rt = prog.get("root_translation")
    if rt:
        tg += (_axis_vec(rt["axis"]) * build_curve(rt.get("curve", "oscillate"), T,
               float(rt["amplitude_m"]), float(rt.get("reps", 1.0)))[:, None])
    return bone6_new.numpy(), r6, tg


def author_program(seq, instruction, model, effort, full_texture, skin_kw=None):
    """Ask the LLM for a motion program given the asset skeleton + a text instruction."""
    rig, mot = load_asset(seq, full_texture=full_texture, **(skin_kw or {}))
    table = describe_skeleton(rig)
    user = (f"ASSET: {seq}\nSKELETON ({rig['joints'].shape[0]} joints), canonical frame:\n{table}\n\n"
            f"MOTION REQUEST: {instruction}\n\nReturn ONLY the JSON motion program.")
    print(f"[agent] querying {model} (effort={effort}) for '{instruction}' ...", flush=True)
    prog = _extract_json(llm_call(SYSTEM, user, model=model, effort=effort))
    return rig, mot, prog


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq"); ap.add_argument("instruction")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--effort", default="high")
    ap.add_argument("--program", default=None, help="load a saved motion-program JSON instead of calling the LLM")
    ap.add_argument("--dry-run", action="store_true", help="print the motion program and exit (no render)")
    ap.add_argument("--full-texture", action="store_true")
    ap.add_argument("--use-orig-skin", action="store_true", help="use skin_weights_orig (un-optimised)")
    ap.add_argument("--no-clean-skin", action="store_true", help="disable graph-distance skin-leakage cleanup")
    ap.add_argument("--skin-graph-dist", type=int, default=3)
    ap.add_argument("--preview"); ap.add_argument("--export"); ap.add_argument("--save-motion")
    ap.add_argument("--save-program", help="write the JSON motion program here")
    a = ap.parse_args()
    skin_kw = dict(use_orig_skin=a.use_orig_skin, clean_skin=not a.no_clean_skin, max_graph_dist=a.skin_graph_dist)

    if a.program:                                               # re-run a saved/edited program (no LLM)
        rig, mot = load_asset(a.seq, full_texture=a.full_texture, **skin_kw)
        prog = json.load(open(a.program))
    else:
        if not llm_available():
            sys.exit("[agent] LLM not available: set ANTHROPIC_API_KEY, or pass --program a hand-written JSON.")
        rig, mot, prog = author_program(a.seq, a.instruction, a.model, a.effort, a.full_texture, skin_kw)

    print("[agent] motion program:\n" + json.dumps(prog, indent=2))
    if a.save_program:
        json.dump(prog, open(a.save_program, "w"), indent=2); print(f"[agent] saved program -> {a.save_program}")
    if a.dry_run:
        sys.exit(0)

    bone6_new, r6, tg = program_to_motion(prog, rig, mot, T=a.frames)
    verts = pose_sequence(rig, bone6_new, tg=tg)          # bake root motion (hop/etc) into the exported PC
    v0_art = pose_sequence(rig, bone6_new[0:1])[0]
    v0 = pose_sequence(rig, mot["bone6"][0:1])[0]
    print(f"[agent] {a.seq}: bone6_new {bone6_new.shape}; frame0 articulation vs original max-diff "
          f"{np.abs(v0_art-v0).max():.2e} (should be ~0)")
    if a.preview:
        preview_gif(verts, rig["colors"], a.preview, fps=max(a.frames // 3, 1))
    if a.export:
        export_pc(verts, rig["colors"], a.export)
    if a.save_motion:
        np.savez(a.save_motion, bone6=bone6_new, r6=r6, tg=tg, scale=np.float64(mot["scale"]),
                 E_fit=mot["E_fit"], K_norm=mot["K_norm"], W=np.int64(mot["W"]), H=np.int64(mot["H"]),
                 iou=np.zeros(bone6_new.shape[0], "float32"),
                 names=np.array([f"frame_{i:03d}" for i in range(bone6_new.shape[0])]))
        print(f"[agent] saved motion -> {a.save_motion}")
