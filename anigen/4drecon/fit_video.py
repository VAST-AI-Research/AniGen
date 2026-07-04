"""Stage 4: fit the bear's skeleton motion to the DAVIS video (sequential, warm-started).

Per frame t we optimise a global root rigid transform {Rg (6D), tg} and per-bone local
rotations (axis-angle, identity=0), driving the mesh with LBS.  Losses:
    * silhouette  : soft-IoU + blurred-L2 vs the DAVIS mask
    * flow        : rendered prev->cur pixel flow vs RAFT flow (over bear ∩ coverage)
    * temporal    : ||params_t - params_{t-1}||  (warm-start anchor -> smooth motion)
    * reg         : ||axis-angle||  (keep articulation modest, avoid single-view overfit)
Frame 0 is initialised from the refined rigid pose (stage 3); each later frame warm-starts
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
from the previous solved frame.  Memory: DAVIS masks + RAFT flow live on CPU, only the
current frame's tensors touch the GPU.
"""
import argparse
import os

import numpy as np
import torch

from geometry import (rot6d_to_matrix, matrix_to_rot6d, identity_rot6d, apply_similarity, Skeleton)
from renderer import Renderer
from davis import load_davis, compute_raft_flow, davis_paths
from fit_utils import silhouette_loss, mask_iou
from tracks import TrackSupervisor


def huber(x, delta=2.0):
    a = x.abs()
    return torch.where(a < delta, 0.5 * x ** 2, delta * (a - 0.5 * delta))


def bone_angles_deg(R):
    """mean rotation angle (deg) of a stack of rotations [M,3,3]."""
    tr = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1, 3)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--pose0", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--flow_cache", default=None)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--iters0", type=int, default=200)
    ap.add_argument("--w_sil", type=float, default=1.0)
    ap.add_argument("--w_flow", type=float, default=0.02)
    # priors act on SO(3) chordal distances of the PARENT-relative bone rotation (see loop);
    # weights are calibrated to that scale (bone chordal .mean() ~1e-3), not the old 6D scale.
    ap.add_argument("--w_temp_bone", type=float, default=5.0, help="angular-velocity prior (parent frame)")
    ap.add_argument("--w_temp_root", type=float, default=1.0, help="root rot(chordal)+trans(L2) velocity")
    ap.add_argument("--w_accel_bone", type=float, default=40.0,
                    help="angular-acceleration prior in each bone's PARENT frame: penalizes chordal "
                         "distance to the constant-angular-velocity prediction R_pred = dR@R_{t-1}; "
                         "occluded (gradient-free) bones coast at constant angular velocity instead of freezing")
    ap.add_argument("--w_accel_root", type=float, default=3.0, help="root rot(chordal)+trans(L2) acceleration")
    ap.add_argument("--w_reg_bone", type=float, default=0.1)
    ap.add_argument("--tracks", "--spatrack", dest="tracks", default=None,
                    help="point tracks npz (CoTracker3 or SpatialTracker V2, auto-detected); "
                         "'' disables track supervision")
    ap.add_argument("--w_track2d", type=float, default=0.005,
                    help="2D reprojection weight (per-point long-range correspondence)")
    ap.add_argument("--w_track3d", type=float, default=10.0,
                    help="3D displacement weight (SpatialTracker only; ignored for 2D-only CoTracker3)")
    args = ap.parse_args()
    dev = "cuda"
    R = f"results/{args.seq}"
    args.rig = args.rig or f"{R}/rig.npz"
    args.pose0 = args.pose0 or f"{R}/pose0.npz"
    args.out = args.out or f"{R}/motion.npz"
    args.flow_cache = args.flow_cache or f"{R}/raft_flow.npy"
    if args.tracks is None:
        args.tracks = f"{R}/cotracker.npz"
    frames_dir, ann_dir = davis_paths(args.seq)

    d = np.load(args.rig)
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32)
    weights = torch.tensor(d["skin_weights"], device=dev, dtype=torch.float32)
    sk = Skeleton(d["joints"], d["parents"], device=dev)
    M = sk.M

    p0 = np.load(args.pose0)
    s = float(p0["scale"])
    W, H = int(p0["W"]), int(p0["H"])
    E = torch.tensor(p0["E_fit"], device=dev, dtype=torch.float32)
    K = torch.tensor(p0["K_norm"], device=dev, dtype=torch.float32)
    r6_root0 = torch.tensor(p0["rot6d"], device=dev, dtype=torch.float32)
    tg0 = torch.tensor(p0["tg"], device=dev, dtype=torch.float32)

    nf = None if args.max_frames == 0 else args.max_frames
    frames, masks_np, names, _ = load_davis(frames_dir, ann_dir, H=H, W=W, n_frames=nf)
    T = len(frames)
    masks = torch.tensor(masks_np, dtype=torch.float32)                # CPU [T,H,W]

    # RAFT flow (t-1 -> t), cached on CPU
    if os.path.exists(args.flow_cache):
        flows_np = np.load(args.flow_cache)
        if flows_np.shape[0] < T - 1:
            flows_np = None
    else:
        flows_np = None
    if flows_np is None:
        print("computing RAFT flow ...")
        flows_np = compute_raft_flow(frames, device=dev)               # [T-1,H,W,2]
        np.save(args.flow_cache, flows_np)
    flows = torch.tensor(flows_np, dtype=torch.float32)                # CPU
    torch.cuda.empty_cache()

    r = Renderer(dev)
    id6 = identity_rot6d(M, device=dev)                                # [M,6] bone identity

    track = None
    if args.tracks and os.path.exists(args.tracks) and (args.w_track2d > 0 or args.w_track3d > 0):
        track = TrackSupervisor(args.tracks, E, K, W, H, r.glctx, device=dev)
        print(f"track supervision [{track.source}]: {track.T} frames, {track.valid.shape[1]} tracks, "
              f"has3d={track.has3d} (bind after frame 0)")

    # parameter stores
    BONE6 = torch.zeros(T, M, 6)
    R6 = torch.zeros(T, 6)
    TG = torch.zeros(T, 3)
    ious = np.zeros(T, dtype=np.float32)

    bone_prev = id6.clone()
    r6_prev = r6_root0.clone()
    tg_prev = tg0.clone()
    bone_prev2 = bone_prev.clone()      # two frames ago (for constant-velocity prediction)
    r6_prev2 = r6_prev.clone()
    tg_prev2 = tg_prev.clone()
    vworld_prev = None

    I3 = torch.eye(3, device=dev)

    for t in range(T):
        target = masks[t].to(dev)
        have_accel = (t >= 2) and (args.w_accel_bone > 0 or args.w_accel_root > 0)

        # Rotations are regularized in SO(3), in each joint's PARENT-relative frame:
        #   R_local[j] (= bone[j]) is already the rotation of j relative to its parent.
        #   parent-frame angular increment  dR = R_{t-1} R_{t-2}^T   (left-mult = parent frame)
        #   constant-angular-velocity prediction  R_pred = dR R_{t-1}
        # velocity/accel are the (smooth chordal) SO(3) distances of R_t to R_{t-1} / R_pred.
        Rb_prev = rot6d_to_matrix(bone_prev)                          # [M,3,3]
        Rb_prev2 = rot6d_to_matrix(bone_prev2)
        Rb_pred = (Rb_prev @ Rb_prev2.transpose(-1, -2)) @ Rb_prev    # [M,3,3] const-ang-vel
        Rr_prev = rot6d_to_matrix(r6_prev)                            # [3,3] root (parent = world)
        Rr_prev2 = rot6d_to_matrix(r6_prev2)
        Rr_pred = (Rr_prev @ Rr_prev2.T) @ Rr_prev
        tg_anchor = tg_prev.detach().clone()
        tg_pred = (2 * tg_prev - tg_prev2).detach()                   # translation is a vector space

        # warm-start from the constant-angular-velocity extrapolation so occluded parts start
        # already advanced (rather than at the previous, soon-to-be-frozen, position)
        bone = (matrix_to_rot6d(Rb_pred) if have_accel else bone_prev).clone().requires_grad_(True)
        r6 = (matrix_to_rot6d(Rr_pred) if have_accel else r6_prev).clone().requires_grad_(True)
        tg = (tg_pred if have_accel else tg_prev).clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [bone], "lr": 0.02},
                                {"params": [r6], "lr": 0.01},
                                {"params": [tg], "lr": 0.01}])
        iters = args.iters0 if t == 0 else args.iters
        flow_t = flows[t - 1].to(dev) if t > 0 else None
        mask_prev = masks[t - 1].to(dev) if t > 0 else None

        for it in range(iters):
            R_local = rot6d_to_matrix(bone)                           # [M,3,3] parent-relative
            v_can = sk.lbs(verts, weights, R_local)
            R_root = rot6d_to_matrix(r6)
            v_world = apply_similarity(v_can, s, R_root, tg)
            pred = r.render_silhouette(v_world, faces, E, K, H, W, ssaa=1)
            loss = args.w_sil * silhouette_loss(pred, target, w_iou=1.0, w_l2=2.0,
                                                blur=True, ksize=13, sigma=3.0)
            if flow_t is not None:
                pflow, cover = r.render_flow(vworld_prev, v_world, faces, E, K, H, W)
                trust = (cover > 0.5) & (mask_prev > 0.5)
                if trust.any():
                    diff = (pflow - flow_t)[trust]
                    loss = loss + args.w_flow * huber(diff).mean()
            # velocity prior: SO(3) chordal distance of the parent-relative rotation to prev frame
            loss = loss + args.w_temp_bone * ((R_local - Rb_prev) ** 2).mean()
            loss = loss + args.w_temp_root * (((R_root - Rr_prev) ** 2).mean() + ((tg - tg_anchor) ** 2).mean())
            # acceleration prior: chordal distance to the constant-angular-velocity prediction
            # (parent frame) -> penalizes change of angular velocity; occluded bones coast forward
            if have_accel:
                loss = loss + args.w_accel_bone * ((R_local - Rb_pred) ** 2).mean()
                loss = loss + args.w_accel_root * (((R_root - Rr_pred) ** 2).mean() + ((tg - tg_pred) ** 2).mean())
            # regularization toward identity (chordal)
            loss = loss + args.w_reg_bone * ((R_local - I3) ** 2).mean()
            # SpatialTracker V2 supervision (2D reprojection + 3D camera-frame)
            if track is not None and track.bound:
                tl, _ = track.loss(v_world, t, w2d=args.w_track2d, w3d=args.w_track3d)
                loss = loss + tl
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            R_local = rot6d_to_matrix(bone)
            v_can = sk.lbs(verts, weights, R_local)
            v_world = apply_similarity(v_can, s, rot6d_to_matrix(r6), tg)
            pred = r.render_silhouette(v_world, faces, E, K, H, W, ssaa=1)
            ious[t] = mask_iou(pred, target)
            ang = bone_angles_deg(R_local).item()

        if t == 0 and track is not None:
            nb, resid = track.bind(v_world.detach(), faces)
            print(f"  bound {nb} {track.source} tracks to mesh"
                  + (f"; frame-0 Sim3 residual={resid:.4f}" if track.has3d else ""))

        BONE6[t] = bone.detach().cpu()
        R6[t] = r6.detach().cpu()
        TG[t] = tg.detach().cpu()
        bone_prev2, r6_prev2, tg_prev2 = bone_prev, r6_prev, tg_prev   # shift history
        bone_prev, r6_prev, tg_prev = bone.detach(), r6.detach(), tg.detach()
        vworld_prev = v_world.detach()
        if t % 5 == 0 or t == T - 1:
            print(f"  frame {t:3d}/{T-1} {names[t]}  iou={ious[t]:.3f}  bone_ang={ang:.1f}deg")
        del target, flow_t, mask_prev
        torch.cuda.empty_cache()

    np.savez(args.out,
             bone6=BONE6.numpy().astype(np.float32),
             r6=R6.numpy().astype(np.float32),
             tg=TG.numpy().astype(np.float32),
             scale=s, E_fit=E.cpu().numpy(), K_norm=K.cpu().numpy(),
             W=W, H=H, names=np.array(names), iou=ious)
    print(f"saved -> {args.out}   mean IoU={ious.mean():.3f}  (frames={T})")


if __name__ == "__main__":
    main()
