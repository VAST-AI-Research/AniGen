"""CPU scipy.spatial.cKDTree drop-ins for pytorch3d.ops.knn_points / ball_query.

Matches the pytorch3d return contracts used in AniGen:
  knn_points(p1, p2, K, ...) -> namedtuple(dists[B,P1,K] (squared), idx[B,P1,K], knn)
  ball_query(p1, p2, K, radius, ...) -> (dists[B,P1,K] (squared), idx[B,P1,K] (-1 padded), knn)
Inputs are [B, P, D] tensors on any device; compute on CPU, return on input device.

Confirmed against AniGen call sites:
  - knn_points: attribute access (.idx / .dists) and tuple unpack (dist2, idx, _);
    used with K=1 / K=2 / K=min(8,n); never with K>n at inference, but K>n is
    handled gracefully anyway (cKDTree pads index==n / dist==inf -> we clamp).
  - ball_query: tuple unpack (_, idx, _); idx is -1 padded for out-of-radius /
    out-of-range slots; downstream code filters -1 via .item() in a DSU. Used with
    K=9 radius=threshold and K=max_neighbors+1 (which CAN exceed n).
"""
from collections import namedtuple

import numpy as np
import torch
from scipy.spatial import cKDTree

_KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def knn_points(p1, p2, K=1, return_nn=False, norm=2, **kwargs):
    """Squared-distance KNN. dists are squared euclidean, idx in [0, n)."""
    dev = p1.device
    B = p1.shape[0]
    dists_all, idx_all = [], []
    for b in range(B):
        a = p1[b].detach().cpu().numpy()
        ref = p2[b].detach().cpu().numpy()
        n = ref.shape[0]
        tree = cKDTree(ref)
        d, i = tree.query(a, k=K)
        d = np.atleast_2d(d)
        i = np.atleast_2d(i)
        if K == 1:
            # query returns shape [P] for k=1; reshape to [P, 1]
            d = d.reshape(-1, 1)
            i = i.reshape(-1, 1)
        # K > n: cKDTree pads missing neighbors with index == n and dist == inf.
        # Clamp the index in-range (avoid OOB gather) and zero its distance so it
        # is harmless if ever consumed; valid neighbors are unaffected.
        oob = i >= n
        if oob.any():
            i = i.copy()
            d = d.copy()
            i[oob] = 0
            d[oob] = 0.0
        dists_all.append(torch.from_numpy(np.ascontiguousarray(d)).float().to(dev) ** 2)
        idx_all.append(torch.from_numpy(np.ascontiguousarray(i)).to(dev).long())
    dists = torch.stack(dists_all, 0)
    idx = torch.stack(idx_all, 0)
    knn = None
    if return_nn:
        knn = torch.gather(p2, 1, idx.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1]))
    return _KNN(dists, idx, knn)


def ball_query(p1, p2, K=500, radius=1.0, return_nn=False, **kwargs):
    """Radius-limited KNN. Out-of-radius / out-of-range slots: idx=-1, dist=0."""
    dev = p1.device
    B = p1.shape[0]
    dists_all, idx_all = [], []
    for b in range(B):
        a = p1[b].detach().cpu().numpy()
        ref = p2[b].detach().cpu().numpy()
        n = ref.shape[0]
        tree = cKDTree(ref)
        d, i = tree.query(a, k=K, distance_upper_bound=radius)
        d = np.atleast_2d(d)
        i = np.atleast_2d(i)
        if K == 1:
            d = d.reshape(-1, 1)
            i = i.reshape(-1, 1)
        i_t = torch.from_numpy(np.ascontiguousarray(i)).to(dev).long()
        d_t = torch.from_numpy(np.ascontiguousarray(d)).float().to(dev)
        # cKDTree pads out-of-range (no neighbor within radius / K>n) with
        # index == n and dist == inf. pytorch3d convention: -1 padding, dist 0.
        oob = i_t >= n
        i_t[oob] = -1
        d_t[oob] = 0.0
        idx_all.append(i_t)
        dists_all.append(d_t ** 2)  # pytorch3d returns squared dists
    dists = torch.stack(dists_all, 0)
    idx = torch.stack(idx_all, 0)
    knn = None
    if return_nn:
        # gather requires valid indices; clamp -1 to 0 for the gather then it's
        # the caller's responsibility to mask via idx (pytorch3d does the same).
        safe = idx.clamp_min(0)
        knn = torch.gather(p2, 1, safe.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1]))
    return dists, idx, knn
