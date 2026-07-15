#!/usr/bin/env python3
"""STEP 2 (optional) - fill INTERIOR disocclusion holes in the background point cloud, so the depth /
point-cloud conditioning is complete after removing the original subject.

Pipeline: build bg PC (Pi3 depth, subject removed w/ dilated mask) -> render bg at FRAME0 -> find
INTERIOR holes (connected components with area>=thresh that do NOT touch the image border; small
speckle + edge-touching/external holes are ignored) -> cv2 Telea inpaint of BOTH the RGB and the
metric depth (clean, no hallucination). Then unproject the filled hole pixels -> append to the bg PC
-> save bg_augmented.npz (consumed by compose_and_render).

Skip this step when the subject moves enough that the scene behind it is revealed in other frames
(then there are 0 interior holes and compose_and_render can build the bg directly). Runs in the AniGen
env:
  python fill_bg_holes.py --recon <recon> --mask-dir <recon>/dynamic_mask --dilate 4 --out <dir>
"""
import os, sys, glob, argparse
from os import path, makedirs
import numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vendored _media / _pointcloud

OUT_H, OUT_W = 384, 672


def K4_to_K3(k4):
    K = np.eye(3, dtype=np.float64); K[0, 0], K[1, 1], K[0, 2], K[1, 2] = k4; return K


@torch.no_grad()
def main(a):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = "cuda"
    from _media import load_depths, load_cameras, load_masks, load_video
    from _pointcloud import unproject, render
    import torch.nn.functional as Fnn
    F = a.nf
    makedirs(a.out, exist_ok=True)
    video, _ = load_video(path.join(a.recon, "video.mp4")); video = video[:F]
    depths = load_depths(path.join(a.recon, "depths"))[:F].astype(np.float32)
    c2w, intr = load_cameras(path.join(a.recon, "cameras.npz")); c2w = c2w[:F].astype(np.float64); intr = intr[:F]
    sky = load_masks(path.join(a.recon, "sky_mask"))[:F]
    K3 = np.stack([K4_to_K3(intr[i]) for i in range(F)])
    if a.mask_dir:                                    # recon-format mask dir (already OUT res, e.g. dynamic_mask_both)
        gt = load_masks(a.mask_dir)[:F].astype(bool)
    else:                                             # DAVIS GT pngs (need resize)
        mf = sorted(glob.glob(path.join(a.gt_mask, "*.png")))[:F]
        gt = np.stack([np.asarray(Image.open(f).convert("L").resize((OUT_W, OUT_H), Image.NEAREST)) > 127 for f in mf])

    vid_t = torch.tensor(video/255., dtype=torch.float32, device=dev)
    dep_t = torch.tensor(depths, dtype=torch.float32, device=dev)
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=dev)
    K_t = torch.tensor(K3, dtype=torch.float32, device=dev)
    finite = torch.isfinite(dep_t) & (dep_t > 0)
    sky_t = torch.tensor(sky, device=dev); gt_t = torch.tensor(gt, device=dev)

    def build_bg(dil_px):
        """bg PC with camel removed using dilation of `dil_px` pixels (k=2*px+1). dil_px=0 -> raw GT mask."""
        if dil_px > 0:
            k = 2 * dil_px + 1
            gd = Fnn.max_pool2d(gt_t[:, None].float(), k, stride=1, padding=dil_px)[:, 0] > 0.5
        else:
            gd = gt_t
        st = finite & (~sky_t) & (~gd)
        rgb, xyz, _, _ = unproject(video=vid_t, depths=dep_t, cam_c2w=c2w_t, K=K_t,
                                   dynamic_mask=torch.zeros_like(st), static_mask=st)
        return rgb, xyz

    def render0(rgb, xyz):
        vis = torch.ones(xyz.shape[0], 1, dtype=torch.bool, device=dev)
        v, d, alp, _ = render(points_color=rgb, points_pos=xyz, visible=vis, cam_c2w=c2w_t[0:1], K=K_t[0:1],
                              height=OUT_H, width=OUT_W, dynamic_mask=None, verbose=False)
        return ((v[0].clamp(0, 1).cpu().numpy()*255).astype(np.uint8),
                (alp[0].float() > 0.5).cpu().numpy(), d[0].float().cpu().numpy())

    # ---- SWEEP mode: compare hole size across dilation levels, no fill ----
    if a.sweep:
        levels = [0, 2, 4, 6, 10]
        H, W = OUT_H, OUT_W; area_th = int(a.area_frac * H * W)
        rows = []
        for dp in levels:
            rgb, xyz = build_bg(dp)
            r0, al0, _ = render0(rgb, xyz)
            holes = (~al0).astype(np.uint8)
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
            interior = np.zeros_like(holes, bool); n_int = 0; areas = []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area >= area_th and not (x == 0 or y == 0 or x+w >= W or y+h >= H):
                    interior |= (lbl == i); n_int += 1; areas.append(int(area))
            ov = r0.copy(); ov[interior] = (0.45*ov[interior] + 0.55*np.array([255,0,0])).astype(np.uint8)
            band = np.zeros((24, 2*W, 3), np.uint8)
            cv2.putText(band, f"dilate={dp}px  interior_holes={n_int}  area={sorted(areas,reverse=True)}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
            rows.append(np.concatenate([band, np.concatenate([r0, ov], 1)], 0))
        Image.fromarray(np.concatenate(rows, 0)).save(path.join(a.out, "dilate_sweep.png"))
        print(f"[sweep] -> {a.out}/dilate_sweep.png  (left=bg render, right=interior-hole overlay in red)")
        return

    bg_rgb, bg_xyz = build_bg(a.dilate)
    print(f"[fill] bg PC {bg_xyz.shape[0]} pts (dilate={a.dilate}px)")
    rgb0, alpha0, depth0_pi3 = render0(bg_rgb, bg_xyz)

    # INTERIOR holes: connected comps of ~alpha, area>=thresh, not touching border
    holes = (~alpha0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    fill = np.zeros_like(holes, bool)
    H, W = holes.shape; area_th = int(a.area_frac * H * W)
    kept = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0 or y == 0 or x + w >= W or y + h >= H)
        if area >= area_th and not touches_border:
            fill |= (lbl == i); kept.append((i, int(area)))
    print(f"[fill] holes: {n-1} comps, kept {len(kept)} interior (area>={area_th}px): {[a for _,a in kept]}")
    if not kept:
        print("[fill] no interior holes to fill; saving bg unchanged");
        np.savez(path.join(a.out, "bg_augmented.npz"), xyz=bg_xyz.cpu().numpy(), rgb=bg_rgb.cpu().numpy()); return

    # ---- fill the hole. Default = cv2 Telea inpaint (clean, no hallucination). The legacy PixelHacker path
    #      (opt-in) hallucinated a dark fence-tunnel in the hole = the "missing area" artifact. dilate the
    #      fill a small amount so the seam is redrawn. ----
    seam = 3                                              # small seam redraw
    fill_d = cv2.dilate(fill.astype(np.uint8), np.ones((seam, seam), np.uint8)) > 0
    mask_u8 = (fill_d * 255).astype(np.uint8)
    # cv2 Telea inpaint of BOTH the RGB and the Pi3 metric depth (clean, no hallucination).
    # RGB: Telea extends the surrounding fence/ground texture across the hole (plausible).
    inpaint = cv2.inpaint(rgb0, mask_u8, 5, cv2.INPAINT_TELEA)
    # DEPTH: inpaint the Pi3 metric depth over the hole -> stays in Pi3 scale (fence+ground are smooth)
    dvalid = alpha0 & np.isfinite(depth0_pi3) & (depth0_pi3 > 0)
    lo_d, hi_d = np.percentile(depth0_pi3[dvalid], 1), np.percentile(depth0_pi3[dvalid], 99)
    dn = np.clip((depth0_pi3 - lo_d) / (hi_d - lo_d + 1e-6), 0, 1); dn[~dvalid] = 0
    dfill = cv2.inpaint((dn * 255).astype(np.uint8), mask_u8, 5, cv2.INPAINT_TELEA)
    hole_depth = dfill.astype(np.float32) / 255.0 * (hi_d - lo_d) + lo_d
    Image.fromarray(np.concatenate([rgb0, inpaint], 1)).save(path.join(a.out, "inpaint_compare_cv2.png"))
    print(f"[fill] cv2 Telea inpaint (RGB+depth), Pi3 depth range [{lo_d:.2f},{hi_d:.2f}]")

    # unproject the filled hole pixels via frame0 cam
    ys, xs = np.where(fill)
    z = hole_depth[ys, xs]
    K0 = K3[0]
    X = (xs - K0[0, 2]) / K0[0, 0] * z; Y = (ys - K0[1, 2]) / K0[1, 1] * z
    cam_pts = np.stack([X, Y, z], 1)
    world = (cam_pts @ c2w[0][:3, :3].T) + c2w[0][:3, 3]
    new_rgb = inpaint[ys, xs] / 255.0
    aug_xyz = np.concatenate([bg_xyz.cpu().numpy(), world.astype(np.float32)])
    aug_rgb = np.concatenate([bg_rgb.cpu().numpy(), new_rgb.astype(np.float32)])
    np.savez(path.join(a.out, "bg_augmented.npz"), xyz=aug_xyz, rgb=aug_rgb)
    print(f"[fill] added {world.shape[0]} filled pts -> bg {aug_xyz.shape[0]} total -> {a.out}/bg_augmented.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True); ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--mask-dir", default=None, help="recon-format mask dir (OUT res), e.g. recon/dynamic_mask_both")
    ap.add_argument("--out", required=True); ap.add_argument("--nf", type=int, default=49)
    ap.add_argument("--area-frac", type=float, default=0.004, help="min hole area as frac of image (smaller ignored)")
    ap.add_argument("--dilate", type=int, default=4, help="foreground-mask dilation in px (k=2*px+1); 0=raw mask. "
                    "4 default: at recon res the subject depth-halo is ~3-5px, so 2 leaves residual 'ghost' depth; "
                    "4-5 removes it (holes are then cv2-filled).")
    ap.add_argument("--sweep", action="store_true", help="only render frame0 bg at several dilation levels -> dilate_sweep.png")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    main(a)
