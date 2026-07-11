"""Global joint refinement (all frames): per-frame pose + shared skin + per-vertex texture + coarse
corrective shape, by differentiable rendering. Skin/shape are pushed through Laplacian smoothing before
use (smoothing IS the regulariser). Losses: silhouette + photometric + light relaxed-Chamfer. For
low-coverage results, also fits a global colour MLP (see below). Writes rig_fit.npz + motion_fit.npz.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np, torch
import nvdiffrast.torch as dr
from geometry import rot6d_to_matrix, matrix_to_rot6d, apply_similarity, Skeleton, intrinsics_to_projection
from renderer import Renderer
from davis import load_davis, davis_paths
from fit_utils import soft_iou_loss, mask_iou


def logit(p, eps=1e-3):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="unitree_as2_1")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--motion", default=None, help="input poses (default results/<seq>/motion_loma.npz)")
    ap.add_argument("--out_rig", default=None)
    ap.add_argument("--out_motion", default=None)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--skin_res_iters", type=int, default=80, help="spatial smoothing iters on the (base+residual) skin logits before softmax (aggressive activation)")
    ap.add_argument("--skin_res_scale", type=float, default=0.5, help="bound on the per-logit skin residual (tanh*scale) so it stays in range and can't remain noisy")
    ap.add_argument("--lap_iters", type=int, default=60, help="(unused; legacy) spatial smoothing iters")
    ap.add_argument("--lap_lambda", type=float, default=0.1, help="per-iter smoothing rate; GENTLE (many iters kill burrs, preserve structure)")
    ap.add_argument("--knn", type=int, default=8, help="spatial k-NN for the smoothing graph (couples disconnected mesh components)")
    ap.add_argument("--smooth_time", type=float, default=0.0, help="final temporal smoothing (sigma, frames) of bone6+r6+tg -> de-jitter high-fps clips")
    ap.add_argument("--smooth_kv", type=float, default=1.5, help="edge-preservation of the temporal smoother: small keeps fast turns (camel); LARGE (~8) = near-Gaussian, smooths everything (smooth walks like bear)")
    ap.add_argument("--shape", type=int, default=1, help="1 = also optimise a corrective per-vertex SHAPE offset (thin the legs / round feet less)")
    ap.add_argument("--lr_shape", type=float, default=0.0004)
    ap.add_argument("--shape_lap_iters", type=int, default=100, help="spatial smoothing iters on the shape offset before use (activation; more = smoother)")
    ap.add_argument("--shape_max", type=float, default=0.05, help="per-vertex offset magnitude bound (activation) -> stops disconnected floaters drifting into fragments")
    ap.add_argument("--w_shape_mag", type=float, default=15.0, help="magnitude penalty keeping the corrective shape small")
    ap.add_argument("--w_skin_anchor", type=float, default=0.5, help="L1 anchor of the optimised skin to the initial AniGen skin (keeps skin near the prior)")
    ap.add_argument("--tex_cov_thresh", type=float, default=0.70, help="if the texture-optimised coverage of the mesh is below this, renders fall back to the original AniGen texture")
    ap.add_argument("--tex_cov_cos", type=float, default=0.5, help="min |cos(normal,view)| for a face to count as reliably texture-observed (0.5 = within 60deg; excludes grazing views)")
    ap.add_argument("--colormap_iters", type=int, default=600, help="if coverage < tex_cov_thresh, fit a global MLP colour map on the original texture vs the video (0 disables) -> vertex_colors_global")
    ap.add_argument("--lr_pose", type=float, default=0.001,
                    help="pose is co-optimised with skin (small LR + strong temporal reg keep it from drifting)")
    ap.add_argument("--lr_skin", type=float, default=0.02)
    ap.add_argument("--lr_tex", type=float, default=0.06)
    ap.add_argument("--w_sil", type=float, default=2.0)
    ap.add_argument("--w_photo", type=float, default=0.4)
    ap.add_argument("--w_cham", type=float, default=6.0)
    ap.add_argument("--w_temp", type=float, default=12.0, help="pose velocity/accel smoothness (strong: couples frames, stops per-frame drift under joint opt)")
    ap.add_argument("--w_tex_lap", type=float, default=0.3)
    ap.add_argument("--photo_warmup", type=float, default=0.3, help="fraction of iters before photometric reaches full weight")
    ap.add_argument("--n_gt", type=int, default=1000)
    ap.add_argument("--n_mesh", type=int, default=1000)
    ap.add_argument("--n0", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"; Rd = f"results/{args.seq}"
    args.rig = args.rig or f"{Rd}/rig.npz"
    args.motion = args.motion or f"{Rd}/motion_loma.npz"
    args.out_rig = args.out_rig or f"{Rd}/rig_fit.npz"
    args.out_motion = args.out_motion or f"{Rd}/motion_fit.npz"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    g = torch.Generator(device="cpu").manual_seed(args.seed); gd = torch.Generator(device=dev).manual_seed(args.seed)

    d = dict(np.load(args.rig))
    verts = torch.tensor(d["vertices"], device=dev, dtype=torch.float32); V = verts.shape[0]
    faces = torch.tensor(d["faces"], device=dev, dtype=torch.int32); faces_l = faces.long()
    skin0 = torch.tensor(d["skin_weights"], device=dev, dtype=torch.float32)
    colors0 = torch.tensor(d["vertex_colors"], device=dev, dtype=torch.float32).clamp(0, 1)
    sk = Skeleton(d["joints"], d["parents"], device=dev)
    m = np.load(args.motion, allow_pickle=True)
    s = float(m["scale"]); W, H = int(m["W"]), int(m["H"])
    E = torch.tensor(m["E_fit"], device=dev, dtype=torch.float32); K = torch.tensor(m["K_norm"], device=dev, dtype=torch.float32)
    T = int(m["bone6"].shape[0])
    full = intrinsics_to_projection(K, 0.01, 100.0) @ E
    fd, ad = davis_paths(args.seq); frames_np, masks_np, names, _ = load_davis(fd, ad, H=H, W=W, n_frames=T)
    frames_rgb = torch.tensor(frames_np, dtype=torch.float32); masks = torch.tensor(masks_np, dtype=torch.float32)
    r = Renderer(dev); I3 = torch.eye(3, device=dev)

    edges = torch.cat([faces_l[:, [0, 1]], faces_l[:, [1, 2]], faces_l[:, [2, 0]]], 0)
    # Spatial k-NN smoother: the mesh is a soup of disconnected components, so a mesh-edge Laplacian can't
    # couple a floater to the body; a 3D-neighbour graph does -> used for SHAPE so floaters move coherently.
    from scipy.spatial import cKDTree
    vnp = verts.detach().cpu().numpy()
    _, nbr = cKDTree(vnp).query(vnp, k=args.knn + 1)
    nbr = nbr[:, 1:]
    rows = np.repeat(np.arange(V), args.knn); cols = nbr.reshape(-1)
    ii = np.concatenate([rows, cols]); jj = np.concatenate([cols, rows])
    Asp = torch.sparse_coo_tensor(np.stack([ii, jj]), np.ones(len(ii), np.float32), (V, V)).coalesce().to(dev)
    degsp = torch.sparse.sum(Asp, 1).to_dense().clamp(min=1)[:, None]

    def smooth(x, iters, lam):
        for _ in range(iters):
            x = (1 - lam) * x + lam * (torch.sparse.mm(Asp, x) / degsp)
        return x

    # Mesh-edge smoother on the (possibly clean_faces-cut) topology: used for SKIN so smoothing never
    # crosses a cut -- a severed limb keeps its own skin instead of re-mixing with a spatially-near part.
    me_i = torch.cat([edges[:, 0], edges[:, 1]]); me_j = torch.cat([edges[:, 1], edges[:, 0]])
    A_mesh = torch.sparse_coo_tensor(torch.stack([me_i, me_j]),
                                     torch.ones(me_i.numel(), device=dev), (V, V)).coalesce()
    deg_mesh = torch.sparse.sum(A_mesh, 1).to_dense().clamp(min=1)[:, None]

    def smooth_mesh(x, iters, lam):
        for _ in range(iters):
            x = (1 - lam) * x + lam * (torch.sparse.mm(A_mesh, x) / deg_mesh)
        return x

    logit0 = torch.log(skin0.clamp_min(1e-6))

    def skin_used(skin_res):
        # skin = softmax( log(skin_ori) + smooth_mesh(tanh(res)) * scale ); smoothing IS the regulariser.
        r = smooth_mesh(torch.tanh(skin_res), args.skin_res_iters, args.lap_lambda) * args.skin_res_scale
        return torch.softmax(logit0 + r, dim=1)

    def shape_used(shape_param):
        # corrective offset = spatially-smoothed dV (couples floaters, low-frequency), magnitude-bounded.
        dv = smooth(shape_param, args.shape_lap_iters, args.lap_lambda)
        norm = dv.norm(dim=1, keepdim=True)
        scale = (args.shape_max / norm.clamp(min=1e-6)).clamp(max=1.0)
        return dv * scale

    def proj(p):
        vh = torch.cat([p, torch.ones_like(p[:, :1])], -1); c = vh @ full.T
        return torch.stack([0.5 + 0.5 * c[:, 0] / c[:, 3].clamp(min=1e-6), 0.5 + 0.5 * c[:, 1] / c[:, 3].clamp(min=1e-6)], -1)

    bone6 = torch.tensor(m["bone6"], device=dev, dtype=torch.float32).clone().requires_grad_(True)
    r6 = torch.tensor(m["r6"], device=dev, dtype=torch.float32).clone().requires_grad_(True)
    tg = torch.tensor(m["tg"], device=dev, dtype=torch.float32).clone().requires_grad_(True)
    skin_res = torch.zeros_like(skin0).requires_grad_(True)      # skin RESIDUAL (logit space), init 0 = original
    tex_param = logit(colors0).clone().requires_grad_(True)
    shape_param = torch.zeros_like(verts).requires_grad_(args.shape > 0)
    groups = [
        {"params": [bone6, r6, tg], "lr": args.lr_pose},        # pose co-optimised with skin (enables motions)
        {"params": [skin_res], "lr": args.lr_skin},
        {"params": [tex_param], "lr": args.lr_tex},
    ]
    if args.shape:
        groups.append({"params": [shape_param], "lr": args.lr_shape})
    opt = torch.optim.Adam(groups)
    # precompute GT contour sample sets per frame (for chamfer)
    gt_pts = []
    for t in range(T):
        ys, xs = np.where(masks_np[t] > 0.5)
        if len(ys) == 0:
            gt_pts.append(None); continue
        ss = np.random.default_rng(t).choice(len(ys), size=min(len(ys), args.n_gt), replace=len(ys) < args.n_gt)
        gt_pts.append(torch.tensor(np.stack([xs[ss] / W, ys[ss] / H], 1), device=dev, dtype=torch.float32))

    eval_frames = list(range(0, T, max(1, T // 10)))
    def mean_iou(b6, rr, tt, wsk, vin):
        tot = 0.0
        with torch.no_grad():
            for t in eval_frames:
                vw = apply_similarity(sk.lbs(vin, wsk, rot6d_to_matrix(b6[t])), s, rot6d_to_matrix(rr[t]), tt[t])
                tot += float(mask_iou(r.render_silhouette(vw, faces, E, K, H, W, ssaa=1), masks[t].to(dev)))
        return tot / len(eval_frames)

    with torch.no_grad():
        iou_raw = mean_iou(bone6, r6, tg, skin0, verts)
        iou_sm0 = mean_iou(bone6, r6, tg, skin_used(skin_res), verts)
    print(f"joint pose+skin+texture refine: V={V} T={T} iters={args.iters} lap_iters={args.lap_iters} lap_lambda={args.lap_lambda}")
    print(f"  eval IoU (raw skin) {iou_raw:.3f}  (smoothed skin, pre-opt) {iou_sm0:.3f}")
    for it in range(args.iters):
        wphoto = args.w_photo * min(1.0, it / max(1, args.photo_warmup * args.iters))
        Ncham = max(1, int(round(args.n0 * (1 - it / (0.75 * args.iters)))))
        w = skin_used(skin_res)
        colors = torch.sigmoid(tex_param)
        dV = shape_used(shape_param) if args.shape else 0.0
        verts_c = verts + dV                                        # corrective (coarse) canonical shape
        idx = torch.randperm(T, generator=g)[:args.batch].tolist()
        L_sil = L_photo = L_cham = 0.0
        for t in idx:
            vw = apply_similarity(sk.lbs(verts_c, w, rot6d_to_matrix(bone6[t])), s, rot6d_to_matrix(r6[t]), tg[t])
            gt = masks[t].to(dev)
            img, alpha = r.render_color(vw, faces, colors, E, K, H, W, ssaa=args.ssaa, bg=0.0)
            L_sil = L_sil + soft_iou_loss(alpha, gt)
            if wphoto > 0:
                real = frames_rgb[t].to(dev)
                comp = img * alpha[..., None] + real * (1 - alpha[..., None])   # composite over real frame
                wpix = (alpha.detach() * gt)
                L_photo = L_photo + (wpix[..., None] * (comp - real).abs()).sum() / wpix.sum().clamp_min(1.0)
            if args.w_cham > 0 and gt_pts[t] is not None:
                clip = (torch.cat([vw, torch.ones_like(vw[:, :1])], -1) @ full.T)[None].contiguous()
                rast, _ = dr.rasterize(r.glctx, clip, faces, (H, W))
                mm = rast[0, ..., 3] > 0; mys, mxs = torch.where(mm)
                if len(mys):
                    ms = torch.randperm(len(mys), generator=gd, device=dev)[:min(len(mys), args.n_mesh)]
                    tri = faces_l[(rast[0, mys[ms], mxs[ms], 3].long() - 1)]
                    uu = rast[0, mys[ms], mxs[ms], 0]; vv = rast[0, mys[ms], mxs[ms], 1]
                    wS = torch.stack([uu, vv, 1 - uu - vv], -1)
                    mesh2d = proj((wS[:, :, None] * vw[tri]).sum(1))
                    dd = torch.cdist(mesh2d, gt_pts[t]); na = min(Ncham, gt_pts[t].shape[0]); nb = min(Ncham, mesh2d.shape[0])
                    L_cham = L_cham + dd.topk(na, dim=1, largest=False).values.mean() + dd.topk(nb, dim=0, largest=False).values.mean()
        nb = len(idx)
        L_sil /= nb; L_photo = L_photo / nb if wphoto > 0 else 0.0; L_cham = L_cham / nb if args.w_cham > 0 else 0.0
        vel = ((bone6[1:] - bone6[:-1]) ** 2).mean() + ((r6[1:] - r6[:-1]) ** 2).mean()
        acc = ((bone6[2:] - 2 * bone6[1:-1] + bone6[:-2]) ** 2).mean()
        tex_lap = ((colors[edges[:, 0]] - colors[edges[:, 1]]) ** 2).mean()
        shape_mag = (dV ** 2).mean() if args.shape else 0.0        # keep the corrective shape small (coarse)
        skin_anchor = (w - skin0).abs().mean()                     # L1 anchor to the initial AniGen skin
        loss = (args.w_sil * L_sil + wphoto * L_photo + args.w_cham * L_cham + args.w_temp * (vel + acc)
                + args.w_tex_lap * tex_lap + args.w_shape_mag * shape_mag + args.w_skin_anchor * skin_anchor)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"  it {it:4d}/{args.iters}  sil={float(L_sil):.4f} photo={float(L_photo) if wphoto>0 else 0:.4f} "
                  f"cham={float(L_cham) if args.w_cham>0 else 0:.4f}", flush=True)

    # final TEMPORAL smoothing (de-jitter high-fps clips): Gaussian-smooth bone6+r6+tg over time,
    # re-orthonormalising every rotation.  Applied AFTER the joint opt so it controls the final motion.
    if args.smooth_time > 0:
        # edge-preserving temporal smoothing (bilateral): de-jitter slow motion, PRESERVE fast turns.
        from fit_utils import bilateral_time_smooth
        with torch.no_grad():
            b = bilateral_time_smooth(bone6.detach().cpu().numpy(), args.smooth_time, args.smooth_kv)
            bone6.copy_(matrix_to_rot6d(rot6d_to_matrix(torch.tensor(b, device=dev, dtype=torch.float32))))
            r6.copy_(matrix_to_rot6d(rot6d_to_matrix(torch.tensor(
                bilateral_time_smooth(r6.detach().cpu().numpy(), args.smooth_time, args.smooth_kv), device=dev, dtype=torch.float32))))
            tg.copy_(torch.tensor(bilateral_time_smooth(tg.detach().cpu().numpy(), args.smooth_time, args.smooth_kv), device=dev, dtype=torch.float32))
        print(f"  applied edge-preserving temporal smoothing (sigma={args.smooth_time}) to bone6+r6+tg")

    with torch.no_grad():
        wsk = skin_used(skin_res)
        verts_c = verts + (shape_used(shape_param) if args.shape else 0.0)
        iou_end = mean_iou(bone6, r6, tg, wsk, verts_c)
        print(f"  eval IoU (final)  {iou_end:.3f}  (raw {iou_raw:.3f} -> {iou_end:.3f})")
        # texture coverage = fraction of mesh AREA ever observed front-facing (|cos(normal,view)| >=
        # tex_cov_cos) inside GT.  Low coverage -> render uses the global colormap instead of per-vertex.
        import nvdiffrast.torch as _dr
        Fn = faces.shape[0]
        seen_tri = torch.zeros(Fn, dtype=torch.bool, device=dev)
        campos = torch.linalg.inv(E)[:3, 3]
        for t in range(T):
            vw = apply_similarity(sk.lbs(verts_c, wsk, rot6d_to_matrix(bone6[t])), s, rot6d_to_matrix(r6[t]), tg[t])
            clip = (torch.cat([vw, torch.ones_like(vw[:, :1])], -1) @ full.T)[None].contiguous()
            rast, _ = _dr.rasterize(r.glctx, clip, faces, (H, W))
            ok = (rast[0, ..., 3] > 0) & (masks[t].to(dev) > 0.5)
            tid = (rast[0, ..., 3][ok].long() - 1)
            if not len(tid):
                continue
            v0, v1, v2 = vw[faces_l[:, 0]], vw[faces_l[:, 1]], vw[faces_l[:, 2]]
            nrm = torch.cross(v1 - v0, v2 - v0, dim=1); nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cen = (v0 + v1 + v2) / 3; view = campos - cen; view = view / view.norm(dim=1, keepdim=True).clamp(min=1e-8)
            fcos = (nrm * view).sum(1).abs()
            seen_tri[tid[fcos[tid] >= args.tex_cov_cos]] = True
        vv0, vv1, vv2 = verts[faces_l[:, 0]], verts[faces_l[:, 1]], verts[faces_l[:, 2]]
        tri_area = torch.norm(torch.cross(vv1 - vv0, vv2 - vv0, dim=1), dim=1) * 0.5
        tex_coverage = float(tri_area[seen_tri].sum() / tri_area.sum().clamp(min=1e-8))
        print(f"  texture coverage (front-facing area) = {tex_coverage:.3f}  "
              f"({'OPTIMISED' if tex_coverage >= args.tex_cov_thresh else 'FALLBACK to original texture'} at render)")
        w = wsk.cpu().numpy().astype(np.float32)
        colors = torch.sigmoid(tex_param).clamp(0, 1).cpu().numpy().astype(np.float32)
        verts_out = verts_c.cpu().numpy().astype(np.float32)

    # Low coverage: the per-vertex texture only covers the observed part of the surface.  Instead fit a
    # global colour->colour MLP on the complete original AniGen colours vs the video -- it corrects overall
    # tone/lighting and, having no spatial input, extrapolates cleanly to the unseen surface.
    colors_global = None
    if tex_coverage < args.tex_cov_thresh and args.colormap_iters > 0:
        import nvdiffrast.torch as _dr2
        wsk_t = torch.tensor(w, device=dev); vc_t = torch.tensor(verts_out, device=dev)
        colors0_t = torch.tensor(d["vertex_colors"], device=dev, dtype=torch.float32).clamp(0, 1)
        faces_ds = torch.cat([faces, faces[:, [0, 2, 1]]], 0).contiguous()
        rasts = []; reals = []; gts = []
        with torch.no_grad():
            for t in range(0, T, max(1, T // 40)):
                vw = apply_similarity(sk.lbs(vc_t, wsk_t, rot6d_to_matrix(bone6[t])), s, rot6d_to_matrix(r6[t]), tg[t])
                clip = (torch.cat([vw, torch.ones_like(vw[:, :1])], -1) @ full.T)[None].contiguous()
                rst, _ = _dr2.rasterize(r.glctx, clip, faces_ds, (H, W))
                rasts.append(rst); reals.append(frames_rgb[t].to(dev)); gts.append(masks[t].to(dev))
        cm_net = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                     torch.nn.Linear(128, 3)).to(dev)
        torch.nn.init.zeros_(cm_net[-1].weight); torch.nn.init.zeros_(cm_net[-1].bias)
        def cmap(c): return (c + cm_net(c)).clamp(0, 1)
        opt_cm = torch.optim.Adam(cm_net.parameters(), lr=1e-3)
        for _ in range(args.colormap_iters):
            opt_cm.zero_grad(); loss_cm = 0.0; den = 0.0
            c = cmap(colors0_t)
            for rst, real, gt in zip(rasts, reals, gts):
                col, _ = _dr2.interpolate(c[None].contiguous(), rst, faces_ds); al = (rst[..., -1:] > 0).float()
                img = (col * al)[0] + real * (1 - al[0]); wgt = al[0, ..., 0] * gt
                loss_cm = loss_cm + (torch.abs(img - real).sum(-1) * wgt).sum(); den = den + wgt.sum()
            (loss_cm / den.clamp(min=1)).backward(); opt_cm.step()
        colors_global = cmap(colors0_t).detach().cpu().numpy().astype(np.float32)
        print(f"  fitted global MLP colormap (coverage {tex_coverage:.3f} < {args.tex_cov_thresh}) -> vertex_colors_global")

    d["skin_weights_orig"] = d["skin_weights"]; d["skin_weights"] = w
    d["vertex_colors_orig"] = d["vertex_colors"]; d["vertex_colors"] = colors
    if colors_global is not None:
        d["vertex_colors_global"] = colors_global
    d["tex_coverage"] = np.float32(tex_coverage); d["tex_cov_thresh"] = np.float32(args.tex_cov_thresh)
    if args.shape:
        d["vertices_orig"] = d["vertices"]; d["vertices"] = verts_out
    np.savez(args.out_rig, **d)
    np.savez(args.out_motion, bone6=bone6.detach().cpu().numpy().astype(np.float32),
             r6=r6.detach().cpu().numpy().astype(np.float32), tg=tg.detach().cpu().numpy().astype(np.float32),
             scale=s, E_fit=E.cpu().numpy(), K_norm=K.cpu().numpy(), W=W, H=H, names=np.array(names), iou=m["iou"])
    print(f"saved -> {args.out_rig}\nsaved -> {args.out_motion}")


if __name__ == "__main__":
    main()
