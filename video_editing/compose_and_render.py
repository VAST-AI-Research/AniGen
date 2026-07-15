#!/usr/bin/env python3
"""STEP 3 - compose the re-animated asset into the reconstructed scene and render the conditioning.

Scene PC from the recon depth/cameras with the original subject removed via its mask; insert the
re-animated AniGen asset 3D-STATIC (fixed world pos, new articulation only), placed by CAMERA alignment:
asset_world = c2w[0] . depthscale . E_fit . asset_anigen_world, then a mask bbox correction at frame0
(robust to K_norm vs recon-K FOV diff). Camera follows the recon trajectory, OR is locked at frame0
(--static-camera) for in-place motions. Renders bg+asset -> video_pc + depth_video + masks + cameras,
i.e. the conditioning consumed by the VACE generation backend (STEP 4).

Uses the vendored _media / _pointcloud (runs in the AniGen env):
  python compose_and_render.py --recon <recon_dir> --mask-dir <recon>/dynamic_mask \
    --bg-npz <holefill>/bg_augmented.npz --asset-pc <reanim_pc> \
    --motion <AniGen>/results/<asset>/reanim_*.npz --rig <AniGen>/results/<asset>/rig_fit.npz \
    --out <out_dir> [--static-camera]
"""
import os, sys, glob, argparse
from os import makedirs, path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vendored _media / _pointcloud

OUT_H, OUT_W = 384, 672
SRC_H, SRC_W = 256, 448


def K4_to_K3(k4):
    K = np.eye(3, dtype=np.float64); K[0, 0], K[1, 1], K[0, 2], K[1, 2] = k4; return K


def _rot_axis(axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-9); c, s = np.cos(ang), np.sin(ang); x, y, z = axis
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])


@torch.no_grad()
def main(a):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = "cuda"
    from _media import load_depths, load_cameras, load_masks, save_video, save_cameras, save_masks, load_video
    from _pointcloud import unproject, render
    import torch.nn.functional as Fnn
    from geometry import rot6d_to_matrix  # vendored: AniGen's exact 6D->matrix convention

    F = a.nf
    # ---- recon (Pi3) ----
    video, _ = load_video(path.join(a.recon, "video.mp4")); video = video[:F]        # [F,H,W,3] uint8 384x672
    depths = load_depths(path.join(a.recon, "depths"))[:F].astype(np.float32)         # [F,H,W]
    c2w, intr = load_cameras(path.join(a.recon, "cameras.npz")); c2w = c2w[:F].astype(np.float64); intr = intr[:F]
    sky = load_masks(path.join(a.recon, "sky_mask"))[:F]
    K3 = np.stack([K4_to_K3(intr[i]) for i in range(F)])
    print(f"[v2] recon F={F} HxW={video.shape[1:3]} depth[min/med/max]={depths.min():.2f}/{np.median(depths):.2f}/{depths.max():.2f}")

    # ---- camel removal mask: recon mask dir (e.g. dynamic_mask_both = both camels) or DAVIS GT pngs ----
    if a.mask_dir:
        gt = load_masks(a.mask_dir)[:F].astype(bool)
    else:
        mf = sorted(glob.glob(path.join(a.gt_mask, "*.png")))[:F]
        gt = np.stack([np.asarray(Image.open(f).convert("L").resize((OUT_W, OUT_H), Image.NEAREST)) > 127 for f in mf])
    print(f"[v2] camel removal mask mean={gt.mean():.3f}")

    vid_t = torch.tensor(video / 255.0, dtype=torch.float32, device=dev)
    dep_t = torch.tensor(depths, dtype=torch.float32, device=dev)
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=dev)
    K_t = torch.tensor(K3, dtype=torch.float32, device=dev)
    finite = torch.isfinite(dep_t) & (dep_t > 0)
    sky_t = torch.tensor(sky, dtype=torch.bool, device=dev)
    gt_t = torch.tensor(gt, dtype=torch.bool, device=dev)

    # ---- scene PC: load pre-built COMPLETE bg (both camels removed + PixelHacker hole-fill) if given,
    #      else build here (DILATE mask by a.dilate px to avoid silhouette residual) ----
    if a.bg_npz:
        z = np.load(a.bg_npz)
        bg_xyz = torch.tensor(z["xyz"], dtype=torch.float32, device=dev)
        bg_rgb = torch.tensor(z["rgb"], dtype=torch.float32, device=dev)
        print(f"[v2] background PC loaded from {a.bg_npz}: {bg_xyz.shape[0]} pts (hole-filled)")
    else:
        dp = a.dilate; k = 2 * dp + 1
        gt_dil = Fnn.max_pool2d(gt_t[:, None].float(), k, stride=1, padding=dp)[:, 0] > 0.5 if dp > 0 else gt_t
        static = finite & (~sky_t) & (~gt_dil)
        bg_rgb, bg_xyz, _, _ = unproject(video=vid_t, depths=dep_t, cam_c2w=c2w_t, K=K_t,
                                         dynamic_mask=torch.zeros_like(static), static_mask=static)
        print(f"[v2] background PC (camel removed, dilate={dp}px): {bg_xyz.shape[0]} pts")

    # ---- asset canonical PC (DENSIFIED via barycentric face sampling) ----
    apc = sorted(glob.glob(path.join(a.asset_pc, "*.npz")))[:F]
    verts_seq = [np.load(f)["xyz"].astype(np.float64) for f in apc]
    vcol = np.load(apc[0])["rgb"].astype(np.float32) / 255.0
    faces = np.load(a.rig)["faces"].astype(np.int64)
    rng = np.random.default_rng(0); per = a.per_face
    r1 = rng.random((faces.shape[0], per)); r2 = rng.random((faces.shape[0], per)); su = np.sqrt(r1)
    bw = np.stack([1 - su, su * (1 - r2), su * r2], -1)          # [Fn,per,3] barycentric weights
    densify = lambda v: np.einsum('fkb,fbc->fkc', bw, v[faces]).reshape(-1, 3)
    v_can = [densify(v) for v in verts_seq]
    a_rgb = np.einsum('fkb,fbc->fkc', bw, vcol[faces]).reshape(-1, 3).astype(np.float32)
    print(f"[v2] asset densified {verts_seq[0].shape[0]} verts x {faces.shape[0]} faces -> {v_can[0].shape[0]} pts")

    # ---- 6DOF align to FRAME0 using AniGen fit orientation (E_fit) + Pi3 depth/scale ----
    # AniGen's frame0 fit gives the exact starting orientation (det=+1 proper rotation; VERIFIED it maps
    # canonical-up -> scene-up +0.99 and the asset's head end rises -> a correct rear-up, NOT a handstand).
    # Pi3 depth/mask give metric position+scale. Asset stays 3D-static; camera keeps its Pi3 trajectory.
    m = np.load(a.motion, allow_pickle=True)
    E = m["E_fit"].astype(np.float64)
    Rg0 = rot6d_to_matrix(torch.tensor(m["r6"][0])).numpy().astype(np.float64)
    m0 = finite[0] & gt_t[0]
    _, r_xyz, _, _ = unproject(video=vid_t[0:1], depths=dep_t[0:1], cam_c2w=c2w_t[0:1], K=K_t[0:1],
                               dynamic_mask=torch.zeros_like(m0)[None], static_mask=m0[None])
    r_xyz = r_xyz.cpu().numpy()
    up = -c2w[:, :3, 1].mean(0); up = up / np.linalg.norm(up)
    R = c2w[0][:3, :3] @ E[:3, :3] @ Rg0              # canonical -> Pi3 world orientation (AniGen frame0 fit)
    hh = lambda P, u: np.percentile(P @ u, 97) - np.percentile(P @ u, 3)
    c_a = np.median(v_can[0], 0)
    s_al = hh(r_xyz, up) / max(hh((v_can[0] - c_a) @ R.T, up), 1e-6)
    c_r = np.median(r_xyz, 0)
    best = None                                       # E_fit orientation definitive; guard 180-yaw ambiguity by IoU
    for flip in (0.0, np.pi):
        Rf = _rot_axis(up, flip) @ R
        place = (lambda Rf: (lambda vc: s_al * ((vc - c_a) @ Rf.T) + c_r))(Rf)
        aw0 = torch.tensor(place(v_can[0]), dtype=torch.float32, device=dev)
        _, _, alpA, _ = render(points_color=torch.tensor(a_rgb, device=dev), points_pos=aw0,
                               visible=torch.ones(aw0.shape[0], 1, dtype=torch.bool, device=dev),
                               cam_c2w=c2w_t[0:1], K=K_t[0:1], height=OUT_H, width=OUT_W, dynamic_mask=None, verbose=False)
        iou = float((alpA[0].bool() & gt_t[0]).sum()) / float((alpA[0].bool() | gt_t[0]).sum() + 1e-6)
        if best is None or iou > best[0]:
            best = (iou, Rf)
    iou, Rf = best
    # final placement: optional shrink (--asset-scale) + slide the asset along the GROUND toward the camera
    # (--asset-fwd, world units; camera tilts down so this also lowers it in frame), then RE-GROUND so the
    # feet sit on the real subject's ground contact -> the asset always stands on the ground, never floats.
    scale = a.asset_scale * s_al
    fwd = c2w[0][:3, 2]; tocam = -fwd; gdir = tocam - (tocam @ up) * up
    gdir = gdir / (np.linalg.norm(gdir) + 1e-9)        # ground-plane direction toward the camera
    base = lambda vc, extra=np.zeros(3): scale * ((vc - c_a) @ Rf.T) + c_r + extra
    ground_level = np.percentile(r_xyz @ up, 2)        # real subject's feet (ground contact) along up
    feet_now = np.percentile(base(v_can[0]) @ up, 2)
    extra = (ground_level - feet_now) * up + a.asset_fwd * gdir   # re-ground + forward slide
    place = lambda vc: base(vc, extra)
    print(f"[v2] 6DOF-frame0 align (E_fit): scale={s_al:.3f}x{a.asset_scale}={scale:.3f} IoU(frame0)={iou:.3f} "
          f"fwd={a.asset_fwd} reground={float(ground_level - feet_now):.3f}")
    asset_world = [place(vc) for vc in v_can]         # 3D-static placement, per-frame new articulation
    N_a = asset_world[0].shape[0]

    # ---- frame-0 alignment check ----
    aw0 = torch.tensor(asset_world[0], dtype=torch.float32, device=dev)
    col = torch.tensor(a_rgb, dtype=torch.float32, device=dev)
    allp = torch.cat([bg_xyz, aw0]); allc = torch.cat([bg_rgb, col])
    v0, _, alp0, _ = render(points_color=allc, points_pos=allp, visible=torch.ones(allp.shape[0], 1, dtype=torch.bool, device=dev),
                            cam_c2w=c2w_t[0:1], K=K_t[0:1], height=OUT_H, width=OUT_W, dynamic_mask=None, verbose=False)
    v0 = (v0[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # asset-only silhouette IoU vs GT
    _, _, alpA, _ = render(points_color=col, points_pos=aw0, visible=torch.ones(N_a, 1, dtype=torch.bool, device=dev),
                           cam_c2w=c2w_t[0:1], K=K_t[0:1], height=OUT_H, width=OUT_W, dynamic_mask=None, verbose=False)
    aA = alpA[0].bool(); iou = float((aA & gt_t[0]).sum()) / float((aA | gt_t[0]).sum() + 1e-6)
    makedirs(a.out, exist_ok=True)
    Image.fromarray(np.concatenate([video[0], v0], 1)).save(path.join(a.out, "align_check_frame0.png"))
    print(f"[v2] frame0 IoU(asset vs GT)={iou:.3f} -> {a.out}/align_check_frame0.png")
    if a.check_only:
        return

    # ---- render bg + per-frame asset (visible matrix) from Pi3 moving cams ----
    asset_all = np.concatenate(asset_world)           # [F*N_a,3]
    allp = torch.cat([bg_xyz, torch.tensor(asset_all, dtype=torch.float32, device=dev)])
    allc = torch.cat([bg_rgb, torch.tensor(np.tile(a_rgb, (F, 1)), dtype=torch.float32, device=dev)])
    Nbg = bg_xyz.shape[0]
    vis = torch.zeros(allp.shape[0], F, dtype=torch.bool, device=dev); vis[:Nbg] = True
    for t in range(F):
        vis[Nbg + t*N_a: Nbg + (t+1)*N_a, t] = True
    dyn = torch.zeros(allp.shape[0], dtype=torch.bool, device=dev); dyn[Nbg:] = True
    # camera trajectory:
    #   --orbit-camera : keep the asset centered but slowly ORBIT it about world-up (rigidly rotate the
    #                    frame-0 camera around the asset center by an eased azimuth sweep) -> shows a
    #                    viewpoint change while the subject stays framed. Frame0 == recon frame0 camera.
    #   --static-camera: lock at frame0 (in-place motion, original camera dollies away).
    #   default        : follow the Pi3 recon camera (subject itself travels, e.g. the walking camel).
    if a.camera_npy:                                       # custom trajectory (from a .npy or the camera agent)
        z = np.load(a.camera_npy); c2w_c = z["c2w"][:F].astype(np.float64); intr_c = z["intr"][:F].astype(np.float64)
        K_c = np.stack([K4_to_K3(intr_c[i]) for i in range(F)])
        c2w_r = torch.tensor(c2w_c, dtype=torch.float32, device=dev); K_r = torch.tensor(K_c, dtype=torch.float32, device=dev)
        c2w_save, intr_save = c2w_c, intr_c
        src_orig = np.repeat(video[0:1], F, 0)
    elif a.orbit_camera:
        up = -c2w[:, :3, 1].mean(0); up = up / (np.linalg.norm(up) + 1e-9)   # world up (cams ~upright)
        target = np.median(asset_world[0], axis=0)         # asset frame-0 world center = orbit pivot
        R0 = c2w[0][:3, :3]; eye0 = c2w[0][:3, 3]
        tt = np.arange(F) / max(F - 1, 1)
        ease = tt * tt * tt * (tt * (tt * 6 - 15) + 10)    # smootherstep, one-way eased sweep
        az = np.deg2rad(a.orbit_deg) * ease
        c2w_orb = np.repeat(c2w[0:1], F, 0).astype(np.float64)
        for i in range(F):
            Rot = _rot_axis(up, az[i])                      # rigid rotation about the pivot keeps it centered
            c2w_orb[i, :3, :3] = Rot @ R0
            c2w_orb[i, :3, 3] = target + Rot @ (eye0 - target)
        c2w_r = torch.tensor(c2w_orb, dtype=torch.float32, device=dev); K_r = K_t[0:1].repeat(F, 1, 1)
        c2w_save, intr_save = c2w_orb, np.repeat(intr[0:1], F, 0)
        src_orig = np.repeat(video[0:1], F, 0)
    elif a.static_camera:
        c2w_r = c2w_t[0:1].repeat(F, 1, 1); K_r = K_t[0:1].repeat(F, 1, 1)
        c2w_save = np.repeat(c2w[0:1], F, 0); intr_save = np.repeat(intr[0:1], F, 0)
        src_orig = np.repeat(video[0:1], F, 0)
    else:
        c2w_r, K_r = c2w_t, K_t
        c2w_save, intr_save = c2w, intr
        src_orig = video
    vpc, dpc, alpha, dynm = render(points_color=allc, points_pos=allp, visible=vis, cam_c2w=c2w_r, K=K_r,
                                   height=OUT_H, width=OUT_W, dynamic_mask=dyn, verbose=True)  # render at OUT res (K matches)
    vpc_np = (vpc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    alpha_np = (alpha.float() > 0.5).cpu().numpy()
    dyn_np = (dynm.float() > 0.5).cpu().numpy()
    # depth video (normalized) for VACE control
    dep = dpc.float().cpu().numpy()
    m0 = alpha_np
    lo, hi = np.percentile(dep[m0], 2), np.percentile(dep[m0], 98)
    depth_vis = np.clip((dep - lo) / (hi - lo + 1e-6), 0, 1); depth_vis[~m0] = 0.5
    depth_np = (np.stack([depth_vis]*3, -1) * 255).astype(np.uint8)

    save_video(path.join(a.out, "video_pc.mp4"), vpc_np, fps=a.fps, quality=9)
    save_video(path.join(a.out, "depth_video.mp4"), depth_np, fps=a.fps, quality=9)
    save_cameras(path.join(a.out, "cameras_tgt.npz"), c2w_save, intr_save.astype(np.float64))
    save_masks(path.join(a.out, "alpha_mask_pc"), alpha_np)
    save_masks(path.join(a.out, "dynamic_mask_pc"), dyn_np)
    save_masks(path.join(a.out, "alpha_mask_src"), np.ones((F, OUT_H, OUT_W), bool))
    save_masks(path.join(a.out, "dynamic_mask_src"), np.zeros((F, OUT_H, OUT_W), bool))
    save_video(path.join(a.out, "video_src_static.mp4"), np.repeat(video[0:1], F, 0), fps=a.fps, quality=9)
    save_video(path.join(a.out, "video_src_original.mp4"), src_orig, fps=a.fps, quality=9)
    save_video(path.join(a.out, "_preview.mp4"), np.concatenate([video, vpc_np], 2), fps=a.fps, quality=8)
    print(f"[v2] DONE -> {a.out}  (pc-alpha {alpha_np.mean():.2f} dyn {dyn_np.mean():.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True); ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--mask-dir", default=None, help="recon-format removal mask dir (e.g. recon/dynamic_mask_both)")
    ap.add_argument("--bg-npz", default=None, help="pre-built complete bg npz (xyz,rgb) from fill_bg_holes")
    ap.add_argument("--dilate", type=int, default=4, help="foreground-removal-mask dilation px when building bg here (k=2*px+1); 4 default removes the subject depth-halo/ghost")
    ap.add_argument("--asset-pc", required=True); ap.add_argument("--motion", required=True, help="reanimate/blender motion npz (E_fit + r6 give the frame-0 placement)")
    ap.add_argument("--rig", required=True, help="asset rig_fit.npz (its faces densify the point cloud), e.g. results/<asset>/rig_fit.npz")
    ap.add_argument("--per-face", type=int, default=40)
    ap.add_argument("--asset-scale", type=float, default=1.0, help="shrink/grow the placed asset about its center (e.g. 0.8), then re-ground the feet")
    ap.add_argument("--asset-fwd", type=float, default=0.0, help="slide the asset along the ground toward the camera by N world units (stays grounded; lowers it in a down-tilted view)")
    ap.add_argument("--out", required=True); ap.add_argument("--nf", type=int, default=49)
    ap.add_argument("--fps", type=int, default=16); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--static-camera", action="store_true",
                    help="lock the camera at frame0 (locked shot) instead of following the Pi3 trajectory; "
                         "use for IN-PLACE motions (jumping jacks) when the original camera dollies away")
    ap.add_argument("--orbit-camera", action="store_true",
                    help="keep the asset centered but slowly orbit it about world-up (viewpoint change); "
                         "overrides --static-camera")
    ap.add_argument("--orbit-deg", type=float, default=25.0, help="total orbit sweep in degrees (eased, one-way)")
    ap.add_argument("--camera-npy", default=None,
                    help="custom camera trajectory .npy/.npz with c2w[F,4,4] + intr[F,4] (fx,fy,cx,cy at 384x672); "
                         "overrides orbit/static (from a hand-authored file or the camera agent)")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    main(a)
