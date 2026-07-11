"""Sequential per-frame 4D pose fit driven by LoMa appearance correspondences (robust, pick-best).

Per frame (t>=1); frame 0 = the AniGen input pose (already aligned).  For robustness against drift
(a bad frame poisoning the next) every frame:
  1. SCREEN several inits {previous fitted pose, frame-0 pose, their interpolation} with a cheap
     6DoF-rigid + silhouette/relaxed-Chamfer refine, and KEEP THE BEST (so a drifted previous frame
     can always restart from the clean frame-0 articulation).
  2. LoMa + FPS-coverage rounds on the winner: render the textured mesh, match (appearance) vs the
     masked GT with LoMa-G; a geodesic farthest-point set of anchors on the GT mask guarantees EVERY
     far region (each leg/foot, head, tail) owns a correspondence (geodesic-embedding fallback fills
     any anchor LoMa missed).  Pulls are ERROR-weighted (mis-posed limbs pull hardest) and
     REGION-balanced (the textured torso can't out-vote the thin legs).  LoMa re-runs each round.
  3. diff-render hand-off: silhouette + relaxed-N Chamfer snap the contour; LoMa pull kept small.
KEEP-BEST across all stages -> the output is never worse than the best init.  Only the skeleton pose
moves.  Writes a motion.npz-compatible file (default results/<seq>/motion_loma.npz).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np, torch
import nvdiffrast.torch as dr
from geometry import rot6d_to_matrix, matrix_to_rot6d, apply_similarity, Skeleton, intrinsics_to_projection, identity_rot6d
from renderer import Renderer, to_uint8
from davis import load_davis, davis_paths
from fit_utils import soft_iou_loss, mask_iou
from loma_match import load_loma, match_scores as loma_match_scores
from geodesic import geodesic_fps_pixels, geo_nearest, landmark_geo_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="unitree_as2_1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rigid_iters", type=int, default=40)
    ap.add_argument("--screen_iters", type=int, default=45, help="chamfer refine iters when screening each init")
    ap.add_argument("--loma_rounds", type=int, default=2)
    ap.add_argument("--round_iters", type=int, default=55)
    ap.add_argument("--refine_iters", type=int, default=85)
    ap.add_argument("--num_kpts", type=int, default=8192)
    ap.add_argument("--filter_th", type=float, default=0.04)
    ap.add_argument("--drop_pct", type=float, default=98.0)
    ap.add_argument("--min_matches", type=int, default=12, help="skip LoMa pull below this many valid matches")
    ap.add_argument("--n_anchor", type=int, default=18)
    ap.add_argument("--radius", type=float, default=70.0)
    ap.add_argument("--geo_k", type=int, default=3)
    ap.add_argument("--n_mesh", type=int, default=1200)
    ap.add_argument("--n_gt", type=int, default=1200)
    ap.add_argument("--n0", type=int, default=6)
    ap.add_argument("--w_corr", type=float, default=14.0)
    ap.add_argument("--w_corr_refine", type=float, default=3.0)
    ap.add_argument("--w_sil", type=float, default=1.0)
    ap.add_argument("--w_cham", type=float, default=40.0)
    ap.add_argument("--w_temp", type=float, default=4.0)
    ap.add_argument("--w_reg", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_t", type=int, default=0)
    ap.add_argument("--reverse", type=int, default=0, help="1 = anchor at the LAST frame and fit backward (rig/pose0 come from the last frame)")
    ap.add_argument("--smooth_root", type=float, default=0.0, help="post-hoc Gaussian temporal smoothing (sigma, frames) of the root 6DoF r6/tg -- use on high-fps clips (bear/camel)")
    ap.add_argument("--multiview", type=int, default=0, help="1 = spin the mesh to several azimuths, match each vs GT, keep cross-view-verified correspondences (fixes self-occluded limbs)")
    ap.add_argument("--mv_views", default="-30,0,30", help="azimuth offsets (deg) for --multiview")
    ap.add_argument("--mv_bin", type=float, default=16.0)
    ap.add_argument("--mv_agree", type=float, default=28.0, help="GT targets within this across views -> verified (full weight); else single-view (half)")
    args = ap.parse_args()
    dev = "cuda"; Rd = f"results/{args.seq}"; out = args.out or f"{Rd}/motion_loma.npz"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(f"{Rd}/rig.npz")
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32)
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32); facesL = faces.long()
    skin = torch.tensor(d["skin_weights"], device=dev, dtype=torch.float32)
    col = torch.tensor(d["vertex_colors"], device=dev, dtype=torch.float32).clamp(0, 1)
    sk = Skeleton(d["joints"], d["parents"], device=dev)
    # camera + frame-0 rigid root come from refine_pose (pose0.npz); frame-0 articulation = rest pose.
    p0 = np.load(f"{Rd}/pose0.npz", allow_pickle=True)
    s = float(p0["scale"]); W, H = int(p0["W"]), int(p0["H"])
    E = torch.tensor(p0["E_fit"], device=dev, dtype=torch.float32); K = torch.tensor(p0["K_norm"], device=dev, dtype=torch.float32)
    Mb = sk.M
    fd, ad = davis_paths(args.seq); frames, masks, names, _ = load_davis(fd, ad, H=H, W=W, n_frames=None)
    T = len(frames)
    full = intrinsics_to_projection(K, 0.01, 100.0) @ E
    r = Renderer(dev); I3 = torch.eye(3, device=dev); g = torch.Generator(device=dev).manual_seed(args.seed)

    def proj(p):
        vh = torch.cat([p, torch.ones_like(p[:, :1])], -1); c = vh @ full.T
        return torch.stack([0.5 + 0.5 * c[:, 0] / c[:, 3].clamp(min=1e-6), 0.5 + 0.5 * c[:, 1] / c[:, 3].clamp(min=1e-6)], -1)

    def world(bone, r6, tg):
        return apply_similarity(sk.lbs(verts, skin, rot6d_to_matrix(bone)), s, rot6d_to_matrix(r6), tg)

    def rast_of(vw):
        clip = (torch.cat([vw, torch.ones_like(vw[:, :1])], -1) @ full.T)[None].contiguous()
        rast, _ = dr.rasterize(r.glctx, clip, faces, (H, W))
        return rast, clip

    def alpha_of(rast, clip):
        return dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, faces)[0, ..., 0]

    def measure(bone, r6, tg, gt):
        with torch.no_grad():
            return float(mask_iou(r.render_silhouette(world(bone, r6, tg), faces, E, K, H, W, ssaa=1), gt))

    def loma_select(rast, mm_np, gt_np, anchors, pA, pB, err, sc):
        # weight = error-emphasis (mis-posed limbs pull hardest), REGION-BALANCED across the FPS
        # anchors so the thin, sparsely-matched legs are not out-voted by the dense textured torso.
        # (Confidence down-weighting was tested and HURT the hard leg frames -- the low-confidence leg
        #  matches are correct and essential; keep-best guards against any overshoot instead.)
        errw = np.clip(err / max(1e-6, np.median(err)), 0.3, 4.0) if len(err) else np.array([])
        if len(pA):
            anchor_of, dmatch = geo_nearest(gt_np, pB, anchors, args.geo_k)
        else:
            anchor_of = np.array([], int); dmatch = np.array([])
        valid_reg = dmatch <= args.radius
        cnt = np.ones(len(anchors))
        for a in anchor_of[valid_reg]:
            cnt[a] += 1
        rpx = list(pA); tpx = list(pB); wl = list(errw / cnt[anchor_of]) if len(pA) else []
        covered = np.zeros(len(anchors), bool); covered[anchor_of[valid_reg]] = True
        miss = np.where(~covered)[0]
        if len(miss):
            mys, mxs = np.where(mm_np)
            if len(mys):
                selp = np.random.default_rng(args.seed).choice(len(mys), size=min(len(mys), args.n_mesh), replace=len(mys) < args.n_mesh)
                mesh_pool = np.stack([mxs[selp], mys[selp]], 1).astype(np.float64)
                _, g2m = landmark_geo_match(mm_np, gt_np, mesh_pool, anchors[miss], args.geo_k, n_anchor=20)
                if g2m is not None:
                    medw = float(np.median(errw)) if len(errw) else 1.0   # fallback: gentle geometric prior
                    for a_i, mesh_i in zip(miss, g2m):
                        rpx.append(mesh_pool[mesh_i]); tpx.append(anchors[a_i]); wl.append(medw)
        if not rpx:
            return None
        rpx = np.asarray(rpx); tpx = np.asarray(tpx); wl = np.asarray(wl)
        xi = np.clip(rpx[:, 0].round().astype(int), 0, W - 1); yi = np.clip(rpx[:, 1].round().astype(int), 0, H - 1)
        on = mm_np[yi, xi]
        if on.sum() < 3:
            return None
        xi, yi, tpx, wl = xi[on], yi[on], tpx[on], wl[on]
        tri_sel = facesL[(rast[0, yi, xi, 3].long() - 1)]
        uu = rast[0, yi, xi, 0]; vv = rast[0, yi, xi, 1]
        w_sel = torch.stack([uu, vv, 1 - uu - vv], -1)
        tgt = torch.tensor(tpx / [W, H], device=dev, dtype=torch.float32)
        wgt = torch.tensor(wl, device=dev, dtype=torch.float32)
        return tri_sel, w_sel, tgt, wgt

    def match_frame(bone, r6, tg, gtB, gt_np, mm_np):
        with torch.no_grad():
            vw = world(bone, r6, tg)
            renderA = to_uint8(r.render_color(vw, faces, col, E, K, H, W, ssaa=2, bg=0.0)[0])
        pA, pB, sc = loma_match_scores(model, renderA, gtB, dev, num_keypoints=args.num_kpts, filter_threshold=args.filter_th)
        if len(pA) < 4:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        err = np.linalg.norm(pA - pB, axis=1)
        thr = np.percentile(err, args.drop_pct)
        xi = np.clip(pA[:, 0].round().astype(int), 0, W - 1); yi = np.clip(pA[:, 1].round().astype(int), 0, H - 1)
        gxi = np.clip(pB[:, 0].round().astype(int), 0, W - 1); gyi = np.clip(pB[:, 1].round().astype(int), 0, H - 1)
        keep = mm_np[yi, xi] & gt_np[gyi, gxi] & (err <= thr)
        return pA[keep], pB[keep], err[keep], sc[keep]

    def rodrigues(axis, deg):
        a = axis / (axis.norm() + 1e-9); th = np.deg2rad(deg)
        Kx = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], device=dev, dtype=torch.float32)
        return torch.eye(3, device=dev) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)

    def mv_correspondences(bone, r6, tg, gtB, gt_np):
        """Multi-view LoMa: spin the posed mesh to several azimuths, match each rotated render vs GT,
        back-project to mesh MATERIAL POINTS, keep cross-view-verified ones (>=2 views agree)."""
        with torch.no_grad():
            vw = world(bone, r6, tg)
            c = vw.mean(0); up = E[:3, :3][1]
            recs = []   # (tri, u, v, gt_x, gt_y, score, orig_px_x, orig_px_y, view)
            for th in [float(x) for x in args.mv_views.split(",")]:
                vw_r = (vw - c) @ rodrigues(up, th).T + c
                renderA = to_uint8(r.render_color(vw_r, faces, col, E, K, H, W, ssaa=2, bg=0.0)[0])
                rast, _ = rast_of(vw_r); mm_np = (rast[0, ..., 3] > 0).cpu().numpy()
                pA, pB, sc = loma_match_scores(model, renderA, gtB, dev, num_keypoints=args.num_kpts, filter_threshold=args.filter_th)
                if len(pA) == 0:
                    continue
                xi = np.clip(pA[:, 0].round().astype(int), 0, W - 1); yi = np.clip(pA[:, 1].round().astype(int), 0, H - 1)
                gxi = np.clip(pB[:, 0].round().astype(int), 0, W - 1); gyi = np.clip(pB[:, 1].round().astype(int), 0, H - 1)
                keep = mm_np[yi, xi] & gt_np[gyi, gxi]
                if keep.sum() == 0:
                    continue
                xi, yi, pB, sc = xi[keep], yi[keep], pB[keep], sc[keep]
                triL = (rast[0, yi, xi, 3].long() - 1)
                uu = rast[0, yi, xi, 0].cpu().numpy(); vv = rast[0, yi, xi, 1].cpu().numpy()
                tri3 = facesL[triL]
                mp3d = (torch.stack([rast[0, yi, xi, 0], rast[0, yi, xi, 1], 1 - rast[0, yi, xi, 0] - rast[0, yi, xi, 1]], -1)[:, :, None] * vw[tri3]).sum(1)
                op = (proj(mp3d).cpu().numpy() * [W, H])
                triL = triL.cpu().numpy()
                for i in range(len(pB)):
                    recs.append((int(triL[i]), float(uu[i]), float(vv[i]), pB[i, 0], pB[i, 1], float(sc[i]), op[i, 0], op[i, 1], th))
        if len(recs) < 4:
            return None
        # cross-view verification: bin by original-view location; verified if >=2 views agree on GT target
        from collections import defaultdict
        bins = defaultdict(list)
        for rr in recs:
            bins[(int(rr[6] // args.mv_bin), int(rr[7] // args.mv_bin))].append(rr)
        tri_list = []; uv_list = []; tgt_list = []; w_list = []
        for _, lst in bins.items():
            gts = np.array([[x[3], x[4]] for x in lst]); med = np.median(gts, 0)
            agree = [x for x in lst if np.linalg.norm([x[3] - med[0], x[4] - med[1]]) <= args.mv_agree]
            nviews = len(set(x[8] for x in agree))
            base_w = 1.0 if nviews >= 2 else 0.4                       # verified full weight, single-view half
            # pick the representative (highest score among agreeing / all)
            rep = max(agree if agree else lst, key=lambda x: x[5])
            disp = np.linalg.norm([rep[3] - rep[6], rep[4] - rep[7]])  # orig_proj -> GT displacement (error emphasis)
            tri_list.append(rep[0]); uv_list.append((rep[1], rep[2])); tgt_list.append((rep[3], rep[4]))
            w_list.append(base_w * float(np.clip(disp / 30.0, 0.5, 4.0)))
        tri_sel = facesL[torch.tensor(tri_list, device=dev)]
        uv = torch.tensor(uv_list, device=dev, dtype=torch.float32)
        w_sel = torch.stack([uv[:, 0], uv[:, 1], 1 - uv[:, 0] - uv[:, 1]], -1)
        tgt = torch.tensor(np.array(tgt_list) / [W, H], device=dev, dtype=torch.float32)
        wgt = torch.tensor(w_list, device=dev, dtype=torch.float32)
        return tri_sel, w_sel, tgt, wgt

    def rigid(bone, r6, tg, gt, iters):
        r6 = r6.clone().requires_grad_(True); tg = tg.clone().requires_grad_(True); bone = bone.clone()
        opt = torch.optim.Adam([r6, tg], lr=args.lr)
        for it in range(iters):
            vw = world(bone, r6, tg); rast, clip = rast_of(vw)
            loss = soft_iou_loss(alpha_of(rast, clip), gt)
            opt.zero_grad(); loss.backward(); opt.step()
        return bone.detach(), r6.detach(), tg.detach()

    def refine_dr(bone, r6, tg, gt, gt_pts, Rb_prev, iters, sel=None, w_corr=0.0):
        bone = bone.clone().requires_grad_(True); r6 = r6.clone().requires_grad_(True); tg = tg.clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [bone], "lr": args.lr}, {"params": [r6, tg], "lr": args.lr * 0.5}])
        best = (measure(bone, r6, tg, gt), bone.detach().clone(), r6.detach().clone(), tg.detach().clone())
        for it in range(iters):
            N = max(1, int(round(args.n0 * (1 - it / (0.75 * iters)))))
            vw = world(bone, r6, tg); rast_i, clip_i = rast_of(vw)
            mm = rast_i[0, ..., 3] > 0; mys, mxs = torch.where(mm)
            if len(mys) == 0:
                break
            msel = torch.randperm(len(mys), generator=g, device=dev)[:min(len(mys), args.n_mesh)]
            tri = facesL[(rast_i[0, mys[msel], mxs[msel], 3].long() - 1)]
            uu = rast_i[0, mys[msel], mxs[msel], 0]; vv = rast_i[0, mys[msel], mxs[msel], 1]
            wS = torch.stack([uu, vv, 1 - uu - vv], -1)
            mesh2d = proj((wS[:, :, None] * vw[tri]).sum(1))
            d0 = torch.cdist(mesh2d, gt_pts); na = min(N, gt_pts.shape[0]); nb = min(N, mesh2d.shape[0])
            cham = d0.topk(na, dim=1, largest=False).values.mean() + d0.topk(nb, dim=0, largest=False).values.mean()
            sil = soft_iou_loss(alpha_of(rast_i, clip_i), gt)
            temp = ((rot6d_to_matrix(bone) - Rb_prev) ** 2).mean(); reg = ((rot6d_to_matrix(bone) - I3) ** 2).mean()
            loss = args.w_sil * sil + args.w_cham * cham + args.w_temp * temp + args.w_reg * reg
            if sel is not None and w_corr > 0:
                tri_sel, w_sel, tgt, wgt = sel
                mp2d = proj((w_sel[:, :, None] * vw[tri_sel]).sum(1))
                loss = loss + w_corr * (wgt * torch.sqrt(((mp2d - tgt) ** 2).sum(1) + 1e-4)).sum() / wgt.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 15 == 14 or it == iters - 1:
                iou = measure(bone, r6, tg, gt)
                if iou > best[0]:
                    best = (iou, bone.detach().clone(), r6.detach().clone(), tg.detach().clone())
        return best

    def loma_rounds(bone, r6, tg, gt, gt_np, gtB, anchors, Rb_prev):
        best = (measure(bone, r6, tg, gt), bone.clone(), r6.clone(), tg.clone()); sel = None
        bone = bone.clone().requires_grad_(True); r6 = r6.clone().requires_grad_(True); tg = tg.clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [bone], "lr": args.lr}, {"params": [r6, tg], "lr": args.lr * 0.5}])
        nmatch = 0
        for rnd in range(args.loma_rounds):
            with torch.no_grad():
                rast, _ = rast_of(world(bone, r6, tg)); mm_np = (rast[0, ..., 3] > 0).cpu().numpy()
            if args.multiview:
                s_out = mv_correspondences(bone.detach(), r6.detach(), tg.detach(), gtB, gt_np)
                nmatch = 0 if s_out is None else s_out[0].shape[0]
                if s_out is None:
                    continue
            else:
                pA, pB, err, scm = match_frame(bone.detach(), r6.detach(), tg.detach(), gtB, gt_np, mm_np)
                nmatch = len(pA)
                if len(pA) < args.min_matches:
                    continue
                s_out = loma_select(rast, mm_np, gt_np, anchors, pA, pB, err, scm)
            if s_out is None:
                continue
            sel = s_out; tri_sel, w_sel, tgt, wgt = sel
            anneal = 1.0 - rnd / max(1, args.loma_rounds)
            for it in range(args.round_iters):
                vw = world(bone, r6, tg); rast_i, clip_i = rast_of(vw)
                mp2d = proj((w_sel[:, :, None] * vw[tri_sel]).sum(1))
                pull = (wgt * torch.sqrt(((mp2d - tgt) ** 2).sum(1) + 1e-4)).sum() / wgt.sum()
                sil = soft_iou_loss(alpha_of(rast_i, clip_i), gt)
                temp = ((rot6d_to_matrix(bone) - Rb_prev) ** 2).mean(); reg = ((rot6d_to_matrix(bone) - I3) ** 2).mean()
                loss = args.w_corr * (0.4 + 0.6 * anneal) * pull + args.w_sil * sil + args.w_temp * temp + args.w_reg * reg
                opt.zero_grad(); loss.backward(); opt.step()
            iou = measure(bone, r6, tg, gt)
            if iou > best[0]:
                best = (iou, bone.detach().clone(), r6.detach().clone(), tg.detach().clone())
        return best, sel, nmatch

    print("loading LoMa-G ..."); model = load_loma(dev, "G"); print("loaded.")

    def fit_one(t, inits):
        """Screen the inits (rigid + cheap chamfer), keep best; LoMa rounds; diff-render hand-off."""
        gt = torch.tensor(masks[t], device=dev); gt_np = masks[t] > 0.5
        gtB = (frames[t] * masks[t][..., None] * 255).astype(np.uint8)
        gys, gxs = np.where(gt_np)
        gsel = np.random.default_rng(args.seed).choice(len(gys), size=min(len(gys), args.n_gt), replace=len(gys) < args.n_gt)
        gt_pts = torch.tensor(np.stack([gxs[gsel] / W, gys[gsel] / H], 1), device=dev, dtype=torch.float32)
        anchors = geodesic_fps_pixels(gt_np, args.n_anchor, args.geo_k, seed_px=np.array([gxs.mean(), gys.min() + 5]))
        Rb_prev = rot6d_to_matrix(inits[0][0]).detach()             # temporal anchor = warm-start articulation
        screened = []
        for (bi, ri, ti) in inits:
            bi, ri, ti = rigid(bi, ri, ti, gt, args.rigid_iters)
            iou, bb, rr, tt = refine_dr(bi, ri, ti, gt, gt_pts, Rb_prev, args.screen_iters)
            screened.append((iou, bb, rr, tt))
        best = max(screened, key=lambda x: x[0])
        (iou_l, bl, rl, tl), sel, nmatch = loma_rounds(best[1], best[2], best[3], gt, gt_np, gtB, anchors, Rb_prev)
        if iou_l > best[0]:
            best = (iou_l, bl, rl, tl)
        iou_r, br, rr2, tr = refine_dr(best[1], best[2], best[3], gt, gt_pts, Rb_prev, args.refine_iters, sel=sel, w_corr=args.w_corr_refine)
        if iou_r > best[0]:
            best = (iou_r, br, rr2, tr)
        return (*best, nmatch)

    BONE6 = torch.zeros(T, Mb, 6); R6 = torch.zeros(T, 6); TG = torch.zeros(T, 3); ious = np.zeros(T, np.float32)
    b0 = identity_rot6d(Mb, device=dev)                          # rest (AniGen canonical) articulation
    r0 = torch.tensor(p0["rot6d"], device=dev, dtype=torch.float32)
    tg0 = torch.tensor(p0["tg"], device=dev, dtype=torch.float32)
    # processing order: forward from frame 0, or (for --reverse) backward from the LAST frame -- the
    # ANCHOR frame is where the AniGen rig + pose0 come from (frame 0 normally, last frame if reversed).
    order = list(range(T - 1, -1, -1)) if args.reverse else list(range(T))
    if args.max_t > 0:
        order = order[:args.max_t]
    b_prev = r_prev = t_prev = f_anchor = None
    for i, t in enumerate(order):
        if i == 0:
            inits = [(b0, r0, tg0)]                              # anchor frame = rest + refined root (VGGT)
        else:
            inits = [(b_prev, r_prev, t_prev)]
            if i >= 2:                                           # drift restarts from the FITTED anchor articulation
                inits.append((f_anchor, r_prev, t_prev))
                inits.append((0.5 * (f_anchor + b_prev), r_prev, t_prev))
        iou, bone, r6, tg, nmatch = fit_one(t, inits)
        BONE6[t] = bone.cpu(); R6[t] = r6.cpu(); TG[t] = tg.cpu(); ious[t] = iou
        b_prev, r_prev, t_prev = bone, r6, tg
        if i == 0:
            f_anchor = bone
        print(f"  frame {t:3d}/{T-1} {names[t]}  iou={iou:.3f}  (matches~{nmatch})", flush=True)

    if args.smooth_root > 0:
        # 6DoF temporal smoothing for high-fps clips: EDGE-PRESERVING (bilateral) so a smooth walk is
        # de-jittered but a FAST turn is preserved (a plain Gaussian smears the turn -> misalignment).
        from fit_utils import bilateral_time_smooth
        R6 = torch.tensor(bilateral_time_smooth(R6.numpy(), args.smooth_root)).float()
        R6 = matrix_to_rot6d(rot6d_to_matrix(R6.to(dev))).cpu()
        TG = torch.tensor(bilateral_time_smooth(TG.numpy(), args.smooth_root)).float()
        for t in order:                                          # refresh IoU after smoothing
            ious[t] = measure(BONE6[t].to(dev), R6[t].to(dev), TG[t].to(dev), torch.tensor(masks[t], device=dev))
        print(f"  applied 6DoF temporal smoothing (sigma={args.smooth_root}) -> mean IoU {np.mean([ious[t] for t in order]):.3f}")
    np.savez(out, bone6=BONE6.numpy().astype(np.float32), r6=R6.numpy().astype(np.float32),
             tg=TG.numpy().astype(np.float32), scale=s, E_fit=E.cpu().numpy(), K_norm=K.cpu().numpy(),
             W=W, H=H, names=np.array(names), iou=ious)
    fit_iou = float(np.mean([ious[t] for t in order]))
    print(f"saved -> {out}  mean IoU={fit_iou:.3f}  (frames={len(order)})")


if __name__ == "__main__":
    main()
