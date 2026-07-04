"""SpatialTracker V2 track supervision for the fixed-camera motion fit.

At frame 0 we rasterize the fitted mesh and, at each track's frame-0 pixel, read the hit
face + barycentric coords -> a persistent 'virtual vertex' bound to the surface.  Thereafter a
bound point's world position at frame t is the barycentric interpolation of its (deformed)
face vertices.  Two supervision terms (both visibility/confidence-gated):

* 2D reprojection : project the bound point with the *fitting* camera and match the track's
  2D pixel trajectory (FOV-consistent, robust; long-range vs frame-to-frame flow).
* 3D camera-frame : after a frame-0 Sim(3) that aligns SpatialTracker's (near-static) camera
  frame to the fitting camera frame, match the bound point's camera-frame 3D position -> adds
  the depth / out-of-plane signal that silhouettes cannot provide.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import numpy as np
import torch
import nvdiffrast.torch as dr

from geometry import umeyama, intrinsics_to_projection


def _huber(x, delta):
    a = x.abs()
    return torch.where(a < delta, 0.5 * x ** 2, delta * (a - 0.5 * delta))


class TrackSupervisor:
    def __init__(self, npz_path, E_fit, K_norm, W, H, glctx, device="cuda",
                 near=0.01, far=100.0, vis_thr=0.5, conf_thr=0.3):
        d = np.load(npz_path)
        self.device = device
        self.E_fit = torch.as_tensor(E_fit, device=device, dtype=torch.float32)
        self.K = torch.as_tensor(K_norm, device=device, dtype=torch.float32)
        self.W, self.H, self.near, self.far = W, H, near, far
        self.glctx = glctx
        H_src, W_src = [int(x) for x in d["hw"]]
        self.frame_idx = [int(x) for x in d["frame_idx"]]        # DAVIS idx per track frame
        self.row_of_davis = {dv: i for i, dv in enumerate(self.frame_idx)}

        if "world_tracks" in d.files:
            # ---- SpatialTracker V2: has 3D (world tracks + per-frame camera) ----
            self.source = "spatracker"
            self.has3d = True
            wt = torch.tensor(d["world_tracks"], device=device, dtype=torch.float32)   # [T,N,3]
            c2w = torch.tensor(d["c2w"], device=device, dtype=torch.float32)
            Kst = torch.tensor(d["K"], device=device, dtype=torch.float32)
            vis = torch.tensor(d["vis"], device=device, dtype=torch.float32)
            conf = torch.tensor(d["conf"], device=device, dtype=torch.float32)
            self.T = wt.shape[0]
            w2c = torch.linalg.inv(c2w)
            self.cam = torch.einsum("tij,tnj->tni", w2c[:, :3, :3], wt) + w2c[:, :3, 3][:, None]
            uvw = torch.einsum("tij,tnj->tni", Kst, self.cam)
            uv = uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-6)     # source-res pixels
            self.valid = (vis > vis_thr) & (conf > conf_thr)
        else:
            # ---- CoTracker3: 2D only (state-of-the-art 2D correspondences) ----
            self.source = "cotracker3"
            self.has3d = False
            self.cam = None
            uv = torch.tensor(d["tracks"], device=device, dtype=torch.float32)         # [T,N,2] source-res px
            vis = torch.tensor(d["vis"], device=device, dtype=torch.float32)
            self.T = uv.shape[0]
            self.valid = vis > vis_thr

        # rescale source-res pixels -> fitting working-res pixels (top-left)
        self.uv = torch.stack([uv[..., 0] / W_src * W, uv[..., 1] / H_src * H], dim=-1)  # [T,N,2]
        self.bound = False

    def _full_proj(self):
        return intrinsics_to_projection(self.K, self.near, self.far) @ self.E_fit

    def _bound_points(self, v_world):
        vtri = v_world[self.face_v]                              # [Nb,3,3]
        return (self.bary[..., None] * vtri).sum(1)             # [Nb,3]

    def bind(self, v_world0, faces):
        """faces: [F,3] int32. Bind tracks to faces via frame-0 rasterization at track pixels."""
        full = self._full_proj()
        vh = torch.cat([v_world0, torch.ones_like(v_world0[..., :1])], -1)
        clip = (vh @ full.T)[None].contiguous()                 # [1,V,4]
        rast, _ = dr.rasterize(self.glctx, clip, faces, (self.H, self.W))   # [1,H,W,4]
        uv0 = self.uv[0]
        col = uv0[:, 0].round().long().clamp(0, self.W - 1)
        row = uv0[:, 1].round().long().clamp(0, self.H - 1)
        samp = rast[0, row, col]                                # [N,4]
        tri = samp[:, 3].long()
        hit = (tri > 0) & self.valid[0]
        bnd = torch.nonzero(hit, as_tuple=True)[0]
        self.bnd = bnd
        self.face = (tri[bnd] - 1).clamp(min=0)
        self.face_v = faces.long()[self.face]                   # [Nb,3] vertex indices
        bu, bv = samp[bnd, 0], samp[bnd, 1]
        self.bary = torch.stack([bu, bv, 1 - bu - bv], -1)     # [Nb,3]

        resid = -1.0
        if self.has3d:
            # Sim(3): spatrack camera frame -> fitting camera frame (frame-0 correspondences)
            q0 = self._bound_points(v_world0)                   # [Nb,3] fitting world
            q0_cam = (q0 @ self.E_fit[:3, :3].T) + self.E_fit[:3, 3]
            p0 = self.cam[0][bnd]                               # spatrack camera frame
            s, R, t = umeyama(p0, q0_cam)
            self.sim3 = (s, R, t)
            resid = (s * (p0 @ R.T) + t - q0_cam).norm(dim=-1).mean().item()
            self.q0_cam = q0_cam                                # [Nb,3] fitting camera frame @ t0
            self.cam0 = p0                                      # [Nb,3] spatrack camera frame @ t0
        self.bound = True
        return len(bnd), resid

    def _sim3(self, p):
        s, R, t = self.sim3
        return s * (p @ R.T) + t

    def loss(self, v_world, davis_t, w2d=1.0, w3d=1.0, huber_px=6.0, huber_3d=0.05):
        """davis_t = DAVIS frame index. Returns (loss_tensor, n_valid)."""
        row = self.row_of_davis.get(int(davis_t), None)
        if not self.bound or row is None:
            return v_world.new_zeros(()), 0
        valid = self.valid[row][self.bnd]
        nv = int(valid.sum())
        if nv == 0:
            return v_world.new_zeros(()), 0
        pts = self._bound_points(v_world)                       # [Nb,3]
        loss = v_world.new_zeros(())
        if w2d > 0:
            full = self._full_proj()
            vh = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)
            clip = vh @ full.T
            ndc = clip[..., :2] / clip[..., 3:4].clamp(min=1e-6)
            pred2d = torch.stack([(0.5 + 0.5 * ndc[..., 0]) * self.W,
                                  (0.5 + 0.5 * ndc[..., 1]) * self.H], -1)
            loss = loss + w2d * _huber((pred2d - self.uv[row][self.bnd])[valid], huber_px).mean()
        if w3d > 0 and self.has3d:
            # bias-free 3D *displacement* (relative to frame 0): removes the constant frame-0
            # alignment/FOV bias, isolating motion incl. depth. Under a similarity the offset
            # cancels: sim3(a)-sim3(b) = s*R*(a-b).
            pcam = (pts @ self.E_fit[:3, :3].T) + self.E_fit[:3, 3]
            s, R, _ = self.sim3
            pred_disp = pcam - self.q0_cam
            tgt_disp = s * ((self.cam[row][self.bnd] - self.cam0) @ R.T)
            loss = loss + w3d * _huber((pred_disp - tgt_disp)[valid], huber_3d).mean()
        return loss, nv
