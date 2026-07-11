"""Mask-internal (geodesic) distance helpers on a coarse grid graph.

Correspondence between two silhouettes must follow paths that stay INSIDE the mask (a straight line
would cut across the background gap between limbs and mis-match leg-A to leg-B).  These utilities build
an 8-connected grid graph over a downsampled mask and run Dijkstra for: geodesic nearest-neighbour
matching, geodesic farthest-point sampling (one anchor per far region -- each leg/foot, head, tail),
and a landmark geodesic-embedding matcher (Gromov-Wasserstein-style, intrinsic).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist as _cdist


def coarsen(dom, k):
    h, w = dom.shape
    hc, wc = h // k, w // k
    return dom[:hc * k, :wc * k].reshape(hc, k, wc, k).max((1, 3))


def build_grid_graph(dom):
    """8-connected grid graph over the True pixels of dom -> (csr graph, idx_map[-1 outside], ys, xs)."""
    h, w = dom.shape
    idx = -np.ones((h, w), np.int64)
    ys, xs = np.where(dom)
    n = len(ys); idx[ys, xs] = np.arange(n)
    I, J, Wt = [], [], []
    for dy, dx, wt in [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
                       (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]:
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nid = np.full(n, -1, np.int64)
        nid[ok] = idx[ny[ok], nx[ok]]
        m = nid >= 0
        I.append(np.arange(n)[m]); J.append(nid[m]); Wt.append(np.full(int(m.sum()), wt))
    g = csr_matrix((np.concatenate(Wt), (np.concatenate(I), np.concatenate(J))), shape=(n, n))
    return g, idx, ys, xs


def snap_nodes(idx, dom_c, px, k):
    """px[.,2]=(x,y) full-res -> coarse node ids; snap pixels outside the coarse domain to nearest inside."""
    hc, wc = idx.shape
    cy = np.clip((px[:, 1] / k).astype(int), 0, hc - 1)
    cx = np.clip((px[:, 0] / k).astype(int), 0, wc - 1)
    nid = idx[cy, cx]
    if (nid < 0).any():
        _, (iy, ix) = distance_transform_edt(~dom_c, return_indices=True)
        bad = nid < 0
        nid[bad] = idx[iy[cy[bad], cx[bad]], ix[cy[bad], cx[bad]]]
    return nid


def geo_match(dom_full, A_px, B_px, k):
    """For each A point, the index of the geodesically-nearest B point (path within dom). -1 if disconnected."""
    dom_c = coarsen(dom_full, k)
    g, idx, ys, xs = build_grid_graph(dom_c)
    A_nodes = snap_nodes(idx, dom_c, A_px, k)
    B_nodes = snap_nodes(idx, dom_c, B_px, k)
    node2b = {}
    for b, nd in enumerate(B_nodes):
        node2b.setdefault(int(nd), b)
    uB = np.unique(B_nodes)
    _, _, srcnode = dijkstra(g, directed=False, indices=uB, min_only=True, return_predecessors=True)
    nearest = srcnode[A_nodes]
    return np.array([node2b.get(int(sn), -1) for sn in nearest], np.int64)


def fps(pts, k):
    """Farthest-point sampling (euclidean) -> indices of k spread points."""
    n = len(pts)
    k = min(k, n)
    sel = [0]
    d = np.full(n, np.inf)
    for _ in range(k - 1):
        d = np.minimum(d, ((pts - pts[sel[-1]]) ** 2).sum(1))
        sel.append(int(d.argmax()))
    return np.array(sel)


def geodesic_fps_pixels(dom_full, k, geo_k, seed_px=None):
    """Farthest-point sampling with the MASK-INTERNAL (geodesic) metric -> k anchor pixels that
    cover every far region of the mask (leg tips, head, tail).  Returns anchor px [k,2] (x,y)."""
    dom_c = coarsen(dom_full, geo_k)
    g, idx, ys, xs = build_grid_graph(dom_c)
    n = len(ys)
    if n == 0:
        return np.zeros((0, 2))
    s0 = int(snap_nodes(idx, dom_c, np.asarray(seed_px, float)[None], geo_k)[0]) if seed_px is not None else 0
    sel = [s0]
    dmin = dijkstra(g, directed=False, indices=[s0])[0]
    for _ in range(min(k, n) - 1):
        cand = dmin.copy(); cand[~np.isfinite(cand)] = -1.0
        nxt = int(np.argmax(cand))
        if cand[nxt] <= 0:
            break
        sel.append(nxt)
        dmin = np.minimum(dmin, dijkstra(g, directed=False, indices=[nxt])[0])
    aX = xs[sel] * geo_k + geo_k // 2
    aY = ys[sel] * geo_k + geo_k // 2
    return np.stack([aX, aY], 1).astype(np.float64)


def geo_nearest(dom_full, from_px, to_px, geo_k):
    """For each from-point: (index of geodesically-nearest to-point, geodesic distance in full-res px)."""
    dom_c = coarsen(dom_full, geo_k)
    g, idx, ys, xs = build_grid_graph(dom_c)
    fn = snap_nodes(idx, dom_c, from_px, geo_k)
    tn = snap_nodes(idx, dom_c, to_px, geo_k)
    node2t = {}
    for ti, nd in enumerate(tn):
        node2t.setdefault(int(nd), ti)
    uT = np.unique(tn)
    dist, _, srcnode = dijkstra(g, directed=False, indices=uT, min_only=True, return_predecessors=True)
    idx_near = np.array([node2t.get(int(sn), -1) for sn in srcnode[fn]], np.int64)
    dnear = dist[fn].astype(np.float64) * geo_k
    return idx_near, dnear


def landmark_geo_match(mesh_mask, gt_mask, mesh_px, gt_px, k, n_anchor=24):
    """Landmark-based geodesic-embedding correspondence (Gromov-Wasserstein-style, intrinsic).

    Anchors = points in the mesh∩GT overlap (reliably corresponded -> same body location).  Each point
    is described by its vector of geodesic distances (WITHIN its own mask) to the anchors; matching is
    nearest-neighbour in that descriptor space -> a leg point matches the GT point with the same
    'which-limb + how-far-along-it' signature, regardless of current position.
    Returns (m2g, g2m) index arrays; (None, None) if the overlap is too small to anchor.
    """
    A = coarsen(mesh_mask, k); B = coarsen(gt_mask, k); OV = A & B
    oy, ox = np.where(OV)
    if len(oy) < 4:
        return None, None
    gA, idxA, _, _ = build_grid_graph(A)
    gB, idxB, _, _ = build_grid_graph(B)
    anch = fps(np.stack([ox, oy], 1).astype(np.float64), min(n_anchor, len(oy)))
    aY, aX = oy[anch], ox[anch]
    aA, aB = idxA[aY, aX], idxB[aY, aX]
    DA = dijkstra(gA, directed=False, indices=aA)
    DB = dijkstra(gB, directed=False, indices=aB)
    BIG = 1e4
    DA[~np.isfinite(DA)] = BIG; DB[~np.isfinite(DB)] = BIG
    mnodes = snap_nodes(idxA, A, mesh_px, k)
    gnodes = snap_nodes(idxB, B, gt_px, k)
    fM = np.ascontiguousarray(DA[:, mnodes].T)
    fG = np.ascontiguousarray(DB[:, gnodes].T)
    C = _cdist(fM, fG)
    return C.argmin(1), C.argmin(0)
