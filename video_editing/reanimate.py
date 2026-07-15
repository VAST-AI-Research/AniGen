#!/usr/bin/env python3
"""Re-animate an AniGen video-fit into a NEW motion, starting from the FITTED frame-0 pose.

Core idea (guarantees frame-0 == original):
  R_new[t,j] = R_delta[t,j] @ R0[j],  R0[j] = rot6d_to_matrix(bone6[0,j]),  R_delta[0,j] = I
Author R_delta on chosen joints via axis_angle_to_matrix; root gait via tg/r6.
Then per frame: v_can = Skeleton.lbs(V, W, rot6d_to_matrix(bone6_new[t]))  -> colored point cloud (v_can, colors).

CLI: reanimate.py <seq> <motion> [--preview out.gif] [--export dir] [--save-motion npz]
  <motion> in {camel_rear_up, robot_jumping_jacks}  (curated presets; add your own as small functions).
For a text-driven motion instead of a preset, use agent_motion.py (same output contract).
Runs on CPU (LBS is light); no GPU needed for posing/preview.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vendored geometry.py
import numpy as np, torch
from geometry import Skeleton, rot6d_to_matrix, matrix_to_rot6d, axis_angle_to_matrix

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')  # AniGen/results


def pick_vertex_colors(d):
    """Vertex colours to render: per-vertex texture-optimised colours if coverage >= threshold, else
    (low coverage) the global-colormap colours if present, else the raw original."""
    files = getattr(d, "files", list(d.keys()))
    cov = float(d["tex_coverage"]) if "tex_coverage" in files else 1.0
    th = float(d["tex_cov_thresh"]) if "tex_cov_thresh" in files else 0.70
    if cov < th:
        if "vertex_colors_global" in files:
            return d["vertex_colors_global"]
        if "vertex_colors_orig" in files:
            return d["vertex_colors_orig"]
    return d["vertex_colors"]


def _graph_distances(parents):
    """All-pairs BFS distance on the undirected skeleton graph (parent<->child). [M,M] int."""
    from collections import deque
    M = len(parents); adj = {j: set() for j in range(M)}
    for j in range(M):
        p = int(parents[j])
        if p >= 0:
            adj[j].add(p); adj[p].add(j)
    D = np.full((M, M), 1 << 20, np.int32)
    for s in range(M):
        D[s, s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if D[s, v] > D[s, u] + 1:
                    D[s, v] = D[s, u] + 1; q.append(v)
    return D


def clean_skin_weights(parents, skin, max_graph_dist=3):
    """Kill skinning leakage between distant body parts (e.g. a leg vertex weighted onto a hand joint).
    For each vertex the argmax joint is its DOMINANT joint; drop any weight on joints whose skeleton-graph
    distance from the dominant joint exceeds `max_graph_dist`, then renormalize. Never changes the
    dominant joint. Fixes bad INITIAL skinning (present even in skin_weights_orig, so not an optimization
    artifact). Set max_graph_dist<0 to disable."""
    W = np.asarray(skin, np.float64).copy()
    if max_graph_dist is None or max_graph_dist < 0:
        return W.astype('float32')
    D = _graph_distances(parents)
    keep = D[W.argmax(1)] <= max_graph_dist            # [V,M] bool
    W *= keep
    s = W.sum(1, keepdims=True); s[s < 1e-8] = 1.0
    return (W / s).astype('float32')


def load_asset(seq, full_texture=False, use_orig_skin=False, clean_skin=True, max_graph_dist=3):
    d = np.load(f'{RESULTS}/{seq}/rig_fit.npz')
    m = np.load(f'{RESULTS}/{seq}/motion_fit.npz', allow_pickle=True)
    # full_texture: use the per-vertex texture-optimised colours directly (skip the coverage<thresh
    # -> global-colormap remap that pick_vertex_colors does when tex_coverage is low)
    colors = d['vertex_colors'] if full_texture else pick_vertex_colors(d)
    # skin weights: fitted (default) or the original un-optimised prediction; then graph-distance cleanup
    # (default on) to remove leg<->hand style leakage that exists in the initial skinning.
    skin = d['skin_weights']
    if use_orig_skin and 'skin_weights_orig' in d.files:
        skin = d['skin_weights_orig']
    if clean_skin:
        skin = clean_skin_weights(d['parents'], skin, max_graph_dist)
    rig = dict(joints=d['joints'].astype('float32'), parents=d['parents'],
               vertices=d['vertices'].astype('float32'), skin=np.asarray(skin, 'float32'),
               colors=colors.astype('float32'))
    mot = dict(bone6=m['bone6'].astype('float32'), r6=m['r6'].astype('float32'),
               tg=m['tg'].astype('float32'), scale=float(m['scale']),
               E_fit=m['E_fit'], K_norm=m['K_norm'], W=int(m['W']), H=int(m['H']))
    return rig, mot


def smootherstep(x):  # 0..1 -> 0..1, C2
    x = np.clip(x, 0, 1); return x * x * x * (x * (x * 6 - 15) + 10)


def ramp_hold_return(T, up=0.25, hold=0.35):
    """0 -> 1 (ramp) -> hold -> 0 (return), fractions of T. Length T."""
    t = np.arange(T) / max(T - 1, 1)
    a = smootherstep(t / up)
    down = smootherstep((1 - t) / (1 - up - hold))
    y = np.where(t < up, a, np.where(t < up + hold, 1.0, down))
    return np.clip(y, 0, 1)


def _foot_indices(rig, frac=0.12):
    """Left / right foot vertex indices = the vertices in the bottom `frac` of the rest z-range (the
    ground-contact points), split by lateral (x) sign."""
    V = np.asarray(rig['vertices']); z = V[:, 2]; thr = z.min() + frac * np.ptp(z)
    foot = np.where(z <= thr)[0]
    return foot[V[foot, 0] > 0], foot[V[foot, 0] < 0]


def _axis_angle_np(axis, ang):
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-9); x, y, zc = a; c, s = np.cos(ang), np.sin(ang)
    return np.array([[c+x*x*(1-c), x*y*(1-c)-zc*s, x*zc*(1-c)+y*s],
                     [y*x*(1-c)+zc*s, c+y*y*(1-c), y*zc*(1-c)-x*s],
                     [zc*x*(1-c)-y*s, zc*y*(1-c)+x*s, c+zc*zc*(1-c)]])


def _yaw_fit(src, dst, up):
    """Rigid transform (rotation about `up` + translation) best-mapping points src->dst. Preserves the
    inter-point distances, so it plants the contact feet without shearing them."""
    up = np.asarray(up, float); up = up / (np.linalg.norm(up) + 1e-9)
    cs, cd = src.mean(0), dst.mean(0); S, D = src - cs, dst - cd
    e1 = np.array([1., 0, 0]) - up * up[0]; e1 /= (np.linalg.norm(e1) + 1e-9); e2 = np.cross(up, e1)
    s2 = np.stack([S @ e1, S @ e2], 1); d2 = np.stack([D @ e1, D @ e2], 1)
    ang = np.arctan2((s2[:, 0]*d2[:, 1] - s2[:, 1]*d2[:, 0]).sum(), (s2*d2).sum(1).sum())
    R = _axis_angle_np(up, ang)
    return R, cd - R @ cs


def stabilize_feet(verts, footL, footR, planted, up, anchor_frame=0):
    """Plant the contact feet: for each frame, rigidly re-pose the whole body (yaw+translate) so the
    planted feet return to their `anchor_frame` positions -> zero ground slide. Blended by `planted`
    (0..1) so it fades out when the feet are airborne."""
    a = np.stack([verts[anchor_frame, footL].mean(0), verts[anchor_frame, footR].mean(0)])
    out = verts.copy()
    for t in range(len(verts)):
        w = float(planted[t])
        if w <= 1e-3:
            continue
        cur = np.stack([verts[t, footL].mean(0), verts[t, footR].mean(0)])
        R, tt = _yaw_fit(cur, a, up)
        out[t] = (1 - w) * verts[t] + w * (verts[t] @ R.T + tt)
    return out


def roll_about_feet(verts, angle_curve, axis, footL, footR, anchor_frame=0):
    """Tilt the whole body about the (fixed) foot pivot by `angle_curve` radians about `axis` -> the body
    leans while the planted feet stay put (feet are at the pivot)."""
    pivot = 0.5 * (verts[anchor_frame, footL].mean(0) + verts[anchor_frame, footR].mean(0))
    out = verts.copy()
    for t in range(len(verts)):
        if abs(float(angle_curve[t])) < 1e-4:
            continue
        R = _axis_angle_np(axis, float(angle_curve[t]))
        out[t] = (verts[t] - pivot) @ R.T + pivot
    return out


def ground_feet(verts, foot_idx, up, ref_frame=0):
    """Shift each frame along `up` so its LOWEST foot sits at the reference frame's ground height. Used so
    a hop's landings return exactly to the original foot height (add the hop AFTER this)."""
    up = np.asarray(up, float); up = up / (np.linalg.norm(up) + 1e-9)
    g0 = (verts[ref_frame, foot_idx] @ up).min()
    out = verts.copy()
    for t in range(len(verts)):
        out[t] = verts[t] + (g0 - (verts[t, foot_idx] @ up).min()) * up
    return out


def compose(bone6_0, R_delta):
    """bone6_new[t,j] = matrix_to_rot6d( R_delta[t,j] @ rot6d_to_matrix(bone6_0[j]) )."""
    T, M = R_delta.shape[:2]
    R0 = rot6d_to_matrix(torch.tensor(bone6_0))  # [M,3,3]
    Rn = torch.matmul(R_delta, R0[None])          # [T,M,3,3]
    return matrix_to_rot6d(Rn.reshape(-1, 3, 3)).reshape(T, M, 6)


def frame0_globals(rig, bone6_0):
    """Global joint rotations at frame0 (GR[j]=GR[parent]@R0[j]). Lets us author deltas in a GLOBAL/
    canonical axis regardless of a joint's local rest frame."""
    sk = Skeleton(rig['joints'], rig['parents'], device='cpu')
    R0 = rot6d_to_matrix(torch.tensor(bone6_0))
    GR, _ = sk.forward_kinematics(R0)
    return GR.numpy()  # [M,3,3]


def build_delta_global(T, M, parents, GR0, edits_global):
    """edits_global: list of (joint_idx, GLOBAL_axis[3], angle_curve[T]).
    The effective global rotation applied to joint j's subtree is GR0[parent]@R_delta@GR0[parent]^-1 =
    Rot(GR0[parent]@local_axis). So set local_axis = GR0[parent]^T @ global_axis to rotate about the
    intended CANONICAL axis (z=up, x=lateral, y=fore/aft)."""
    R = torch.eye(3).repeat(T, M, 1, 1)
    for j, gax, curve in edits_global:                    # multiple edits on one joint COMPOSE (accumulate)
        p = int(parents[j])
        Gp = GR0[p] if p >= 0 else np.eye(3)
        lax = Gp.T @ np.asarray(gax, 'float32'); lax = lax / (np.linalg.norm(lax) + 1e-9)
        aa = torch.tensor(lax)[None, :] * torch.tensor(np.asarray(curve, 'float32'))[:, None]
        R[:, j] = torch.matmul(axis_angle_to_matrix(aa), R[:, j])
    return R


# ----------------------------------------------------------------- motions
def camel_rear_up(rig, mot, T=49):
    """A gentle camel rear-up: the front lifts a LITTLE (the head rises but stays in frame), the two
    FRONT paws wave alternately, the HIND paws stay absolutely planted, and the tail swishes. Camel
    canonical z=up, x=lateral, y=fore/aft, HEAD at y<0. Joints: rear-up pitch=15 (spine just forward of
    the hind branch, so the hind stays put); front-paw roots L=6 R=32; tail base=25; hind paws = the
    rear (y>0) low feet. frame0 == original."""
    M = rig['joints'].shape[0]
    GR0 = frame0_globals(rig, mot['bone6'][0])
    t = np.arange(T) / max(T - 1, 1)
    env = ramp_hold_return(T, up=0.28, hold=0.34)          # rear envelope 0..1
    pitch = -0.42 * env                                    # MILD lift about +x (head rises, stays in frame)
    fl = env * 1.05 * np.sin(2 * np.pi * 2.0 * t)          # front-left paw waves (big enough to read as alternating)
    fr = env * 1.05 * np.sin(2 * np.pi * 2.0 * t + np.pi)  # front-right paw, opposite phase -> alternating
    tail = env * 0.55 * np.sin(2 * np.pi * 2.5 * t)        # tail swishes side to side (yaw)
    edits = [(15, [1, 0, 0], pitch), (6, [1, 0, 0], fl), (32, [1, 0, 0], fr), (25, [0, 0, 1], tail)]
    R_delta = build_delta_global(T, M, rig['parents'], GR0, edits)
    bone6_new = compose(mot['bone6'][0], R_delta)
    r6 = np.repeat(mot['r6'][0:1], T, axis=0)
    tg = np.repeat(mot['tg'][0:1], T, axis=0)              # rear in place -> frame0 == original
    # HIND paws absolutely still: foot-lock only the REAR paws (y>0.28 — the front-right paw sits at
    # y~0.11 on this asymmetric skeleton and is meant to wave, so exclude it).
    verts = pose_sequence(rig, bone6_new, tg=tg)
    V = rig['vertices']; z = V[:, 2]; thr = z.min() + 0.15 * np.ptp(z)
    hind = np.where((z <= thr) & (V[:, 1] > 0.28))[0]
    hL, hR = hind[V[hind, 0] > 0], hind[V[hind, 0] < 0]
    verts = stabilize_feet(verts, hL, hR, np.ones(T), np.array([0., 0., 1.]))
    return bone6_new.numpy(), r6, tg, verts


def robot_jumping_jacks(rig, mot, T=49):
    """Two FAST jumping jacks then a lean-left/almost-topple/recover (unitree_g1_1). Canonical z=up,
    x=lateral, y=fore/aft. Joints: L/R shoulder=2/19, L/R hip=3/18. Feet DON'T slide: the leg spread
    (and the arms) only change while the body is AIRBORNE (the ballistic hop lifts the feet); at each
    ground contact the pose is settled so the feet are planted. The closing lean tilts the whole body
    about the (fixed) foot pivot, so the planted feet stay absolutely still. frame0 == original."""
    M = rig['joints'].shape[0]
    GR0 = frame0_globals(rig, mot['bone6'][0])
    zspan = float(np.ptp(rig['vertices'][:, 2]))
    t = np.arange(T) / max(T - 1, 1)
    je = 0.55                                             # 2 jumps occupy [0, je], lean occupies [je, 1]
    pj = np.clip(t / je, 0, 1)                            # jump-phase progress (2 hops: [0,.5] and [.5,1])
    # spread flips 0->1 (jump1, airborne) then 1->0 (jump2, airborne); settled (still) at each landing
    spread = np.where(pj < 0.5, smootherstep(pj / 0.5), smootherstep((1 - pj) / 0.5))
    spread[t > je] = 0.0
    hop = np.zeros(T)                                     # two ballistic arcs, 0 at each landing
    for lo, hi in [(0.0, 0.5), (0.5, 1.0)]:
        m = (pj >= lo) & (pj <= hi) & (t <= je); fr = (pj[m] - lo) / (hi - lo)
        hop[m] = 4 * fr * (1 - fr)
    hop *= 0.18 * zspan                                   # higher -> real airtime; faster (2 jumps in 55%)
    pl = np.clip((t - je) / (1 - je), 0, 1)              # lean-phase progress
    lean = np.minimum(smootherstep(pl / 0.7), smootherstep((1 - pl) / 0.3))   # 0->1 (topple) ->0 (recover)
    A_arm, A_leg = 2.55, 0.42
    arm_env = np.clip(spread + 0.9 * lean, 0, 1)          # arms up during jumps AND raised during the lean
    edits = [(2, [0, 1, 0], +A_arm * arm_env), (19, [0, 1, 0], -A_arm * arm_env),
             (3, [0, 1, 0], +A_leg * spread), (18, [0, 1, 0], -A_leg * spread)]
    R_delta = build_delta_global(T, M, rig['parents'], GR0, edits)
    bone6_new = compose(mot['bone6'][0], R_delta)
    r6 = np.repeat(mot['r6'][0:1], T, axis=0)
    tg = np.repeat(mot['tg'][0:1], T, axis=0).copy()
    # bake geometry: pose (no hop) -> GROUND the feet to the frame0 height (so every landing returns to
    # the original foot height) -> add the ballistic hop in z -> tilt left about the planted feet.
    verts = pose_sequence(rig, bone6_new, tg=tg)
    fL, fR = _foot_indices(rig); foot = np.concatenate([fL, fR])
    verts = ground_feet(verts, foot, [0, 0, 1])
    verts[:, :, 2] += hop[:, None]                        # feet now land at ground0, rise with the hop
    verts = roll_about_feet(verts, -0.6 * lean, [0, 1, 0], fL, fR)   # lean left about the foot pivot
    return bone6_new.numpy(), r6, tg, verts


def robot_combo(rig, mot, T=41):
    """Combo of gestures for unitree_g1_3: hands-on-hips (ramp in and HOLD) -> turn the head ->
    twist the waist -> bend the knees (dip). Canonical z=up, x=lateral, y=fore/aft. Joints (g1_3):
    R/L shoulder=5/21, R/L elbow=3/23, neck=16, waist=14, R/L knee=6/18. Every curve is 0 at t=0 so
    frame0 == original. Gestures are time-windowed so they play in sequence."""
    M = rig['joints'].shape[0]
    GR0 = frame0_globals(rig, mot['bone6'][0])
    t = np.arange(T) / max(T - 1, 1)

    def ramp_in(a, r=0.15):                                # 0 before a, smootherstep 0->1 over r, hold 1
        return smootherstep((t - a) / r)

    def win_osc(a, b, cycles):                            # windowed sine within [a,b], 0 at edges & outside
        p = np.clip((t - a) / (b - a), 0, 1)
        y = np.sin(np.pi * p) * np.sin(2 * np.pi * cycles * p)
        y[(t < a) | (t > b)] = 0.0
        return y

    def win_dip(a, b):                                    # smooth 0->1->0 within [a,b] (ramp-hold-return)
        p = np.clip((t - a) / (b - a), 0, 1)
        y = np.minimum(smootherstep(p / 0.35), smootherstep((1 - p) / 0.35))
        y[(t < a) | (t > b)] = 0.0
        return y

    def win(a, b):                                        # smooth 0->1->0 window (ramp-hold-return)
        p = np.clip((t - a) / (b - a), 0, 1)
        y = np.minimum(smootherstep(p / 0.3), smootherstep((1 - p) / 0.3)); y[(t < a) | (t > b)] = 0.0; return y

    # richer arm sequence: hands-on-hips -> raise & WAVE the left arm -> both arms OUT -> reach forward in the squat
    hips = win(0.03, 0.34)                                # hands to the waist (both), then release
    wave = win(0.34, 0.60)                                # LEFT arm up + waving
    wave_osc = win_osc(0.36, 0.58, 2.5)                  # side-to-side wave
    armsout = win(0.58, 0.80)                             # both arms out to the sides
    head = win_osc(0.10, 0.34, 2.0)                       # turn the head
    waist = win_osc(0.58, 0.80, 1.5)                      # twist the torso
    knee = win_dip(0.80, 0.99)                            # bend both knees (squat)
    edits = [
        # hands on hips: shoulders slightly out, elbows flex so the hands come to the waist
        (5, [0, 1, 0], +0.30 * hips), (21, [0, 1, 0], -0.30 * hips),
        (3, [1, 0, 0], +1.25 * hips), (23, [1, 0, 0], +1.25 * hips),
        # wave the LEFT arm (21) overhead and rock it side to side; left elbow (23) slight bend
        (21, [0, 1, 0], -1.9 * wave), (21, [1, 0, 0], 0.5 * wave_osc), (23, [1, 0, 0], 0.6 * wave),
        # both arms out to the sides
        (5, [0, 1, 0], +1.3 * armsout), (21, [0, 1, 0], -1.3 * armsout),
        # gestures
        (16, [0, 0, 1], 0.6 * head), (14, [0, 0, 1], 0.5 * waist),
        # squat: reach both arms forward + bend both knees
        (5, [1, 0, 0], -0.8 * knee), (21, [1, 0, 0], -0.8 * knee),
        (6, [1, 0, 0], +0.9 * knee), (18, [1, 0, 0], +0.9 * knee),
    ]
    R_delta = build_delta_global(T, M, rig['parents'], GR0, edits)
    bone6_new = compose(mot['bone6'][0], R_delta)
    r6 = np.repeat(mot['r6'][0:1], T, axis=0)
    zspan = float(np.ptp(rig['vertices'][:, 2]))
    tg = np.repeat(mot['tg'][0:1], T, axis=0).copy()
    tg[:, 2] -= 0.12 * zspan * knee                       # hips dip during the squat
    # in-place gesture -> feet planted the WHOLE time: rigidly re-plant both feet every frame (no slide)
    verts = pose_sequence(rig, bone6_new, tg=tg)
    fL, fR = _foot_indices(rig)
    up = np.array([0., 0., 1.])
    verts = stabilize_feet(verts, fL, fR, np.ones(T), up)
    return bone6_new.numpy(), r6, tg, verts


MOTIONS = {'camel_rear_up': camel_rear_up, 'robot_jumping_jacks': robot_jumping_jacks,
           'robot_combo': robot_combo}


def pose_sequence(rig, bone6_new, tg=None):
    sk = Skeleton(rig['joints'], rig['parents'], device='cpu')
    V = torch.tensor(rig['vertices']); W = torch.tensor(rig['skin'])
    out = []
    for t in range(bone6_new.shape[0]):
        v = sk.lbs(V, W, rot6d_to_matrix(torch.tensor(bone6_new[t]))).numpy()
        if tg is not None:
            v = v + tg[t]                      # per-frame global (re-pivot) translation
        out.append(v)
    return np.stack(out)  # [T,V,3]


def export_pc(verts_seq, colors, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rgb = (colors * 255).astype('uint8')
    for t, v in enumerate(verts_seq):
        np.savez(f'{out_dir}/{t:05d}.npz', xyz=v.astype('float32'), rgb=rgb)
    print(f'[export] {len(verts_seq)} frames -> {out_dir} ({verts_seq.shape[1]} pts/frame)')


def preview_gif(verts_seq, colors, out_gif, fps=16, elev_view='side'):
    """Cheap side-view (y=horiz, z=vert) colored scatter per frame -> gif. No GPU."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, imageio
    V = verts_seq
    ymin, ymax = V[..., 1].min(), V[..., 1].max(); zmin, zmax = V[..., 2].min(), V[..., 2].max()
    pad = 0.15 * max(ymax - ymin, zmax - zmin)
    frames = []
    for t, v in enumerate(V):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        order = np.argsort(v[:, 0])   # painter: draw far-x first
        ax.scatter(v[order, 1], v[order, 2], c=np.clip(colors[order], 0, 1), s=3, edgecolors='none')
        ax.set_xlim(ymin - pad, ymax + pad); ax.set_ylim(zmin - pad, zmax + pad)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'frame {t}', fontsize=8)
        fig.tight_layout(pad=0.2); fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(img.copy()); plt.close(fig)
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f'[preview] {out_gif} ({len(frames)} frames)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('seq'); ap.add_argument('motion', choices=list(MOTIONS))
    ap.add_argument('--frames', type=int, default=49)
    ap.add_argument('--preview'); ap.add_argument('--export'); ap.add_argument('--save-motion')
    ap.add_argument('--full-texture', action='store_true', help='use vertex_colors directly (no low-coverage global-colormap remap)')
    ap.add_argument('--use-orig-skin', action='store_true', help='use skin_weights_orig (un-optimised) instead of the fitted weights')
    ap.add_argument('--no-clean-skin', action='store_true', help='disable graph-distance skin-leakage cleanup')
    ap.add_argument('--skin-graph-dist', type=int, default=3, help='max skeleton-graph distance from a vertex dominant joint to keep a weight')
    a = ap.parse_args()
    rig, mot = load_asset(a.seq, full_texture=a.full_texture, use_orig_skin=a.use_orig_skin,
                          clean_skin=not a.no_clean_skin, max_graph_dist=a.skin_graph_dist)
    bone6_new, r6, tg, verts = MOTIONS[a.motion](rig, mot, T=a.frames)
    print(f'[reanimate] {a.seq}/{a.motion}: bone6_new {bone6_new.shape}')
    if verts is None:                                     # motion may pre-bake verts (foot-lock/lean); else pose here
        verts = pose_sequence(rig, bone6_new, tg=tg)      # bake root motion (e.g. the hop) into the exported PC
    v0_art = pose_sequence(rig, bone6_new[0:1])[0]        # articulation-only frame0 (no global) for the sanity check
    v0_orig = pose_sequence(rig, mot['bone6'][0:1])[0]    # original fitted frame-0 articulation
    print(f'[check] frame0 articulation vs original max-diff = {np.abs(v0_art-v0_orig).max():.2e} (should be ~0)')
    if a.preview: preview_gif(verts, rig['colors'], a.preview, fps=a.frames // 3)
    if a.export: export_pc(verts, rig['colors'], a.export)
    if a.save_motion:                                     # motion_fit-format npz -> feed straight to render_results.py
        np.savez(a.save_motion, bone6=bone6_new, r6=r6, tg=tg, scale=np.float64(mot['scale']),
                 E_fit=mot['E_fit'], K_norm=mot['K_norm'], W=np.int64(mot['W']), H=np.int64(mot['H']),
                 iou=np.zeros(bone6_new.shape[0], 'float32'),
                 names=np.array([f'frame_{i:03d}' for i in range(bone6_new.shape[0])]))
        print(f'[save-motion] {a.save_motion}')
