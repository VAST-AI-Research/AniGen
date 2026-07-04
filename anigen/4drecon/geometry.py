"""Geometry core for the bear 4D pipeline.

Conventions
-----------
* World frame = AniGen canonical **Z-up** frame (object centered near origin, ~unit box).
* Cameras are OpenCV world->camera extrinsics (4x4): +X right, +Y down, +Z forward.
* Normalized intrinsics (3x3), consumed by ``intrinsics_to_projection`` exactly like
  AniGen's ``MeshRenderer`` (fx,fy,cx,cy expressed as fractions of the image size).
* Rotations optimized as 6D (Zhou et al. 2019) -> matrix; no gimbal issues, smooth grads.

Skinning (LBS)
--------------
Rest joint positions ``joints[j]`` (=pivots, identity rest rotation, matching AniGen's
translation-only inverse-bind matrices).  Per-joint *local* rotation ``R_local[j]`` is
applied about the joint pivot in its parent frame.  Forward kinematics accumulates:
    GR_j  = GR_parent @ R_local_j                       (global rotation)
    gp_j  = gp_parent + GR_parent @ (p_j - p_parent)    (posed pivot position)
Skinning transform of joint j maps a rest point x -> gp_j + GR_j @ (x - p_j).
Vertex:  v' = sum_j w_{v,j} (gp_j + GR_j (x - p_j)).
A global similarity (scale s, rotation Rg, translation tg) is applied on top for root
motion:  v_final = s * Rg @ v' + tg.
With all R_local = I and (s,Rg,tg)=(1,I,0) this reproduces the rest mesh exactly.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone imports
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Rotation representations
# --------------------------------------------------------------------------- #
def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """[...,6] -> [...,3,3]. Gram-Schmidt on the two 3-vectors (Zhou et al. 2019)."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns are the basis


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """[...,3,3] -> [...,6] (first two columns)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def identity_rot6d(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    r = torch.zeros(n, 6, device=device, dtype=dtype)
    r[:, 0] = 1.0
    r[:, 4] = 1.0
    return r


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """[...,3] -> [...,3,3] via Rodrigues."""
    theta = aa.norm(dim=-1, keepdim=True)
    k = aa / theta.clamp_min(1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz, ky], -1),
        torch.stack([kz, zero, -kx], -1),
        torch.stack([-ky, kx, zero], -1),
    ], -2)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(K.shape)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def quat_to_matrix(q: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    """Quaternion [...,4] -> [...,3,3]. order 'wxyz' or 'xyzw'."""
    q = torch.nn.functional.normalize(q, dim=-1)
    if order == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """Nearest rotation to a 3x3 matrix (SVD, det=+1)."""
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone()
        U[:, -1] = -U[:, -1]
        R = U @ Vh
    return R


def average_rotations(Rs: torch.Tensor) -> torch.Tensor:
    """Chordal L2 mean of a stack of rotations [N,3,3] -> [3,3]."""
    return project_to_so3(Rs.mean(dim=0))


# --------------------------------------------------------------------------- #
# Umeyama similarity alignment (source -> target)
# --------------------------------------------------------------------------- #
def umeyama(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = True):
    """Least-squares similarity s,R,t with dst ~= s R src + t.

    src, dst: [N,3].  Returns (s: scalar tensor, R: [3,3], t: [3]).
    """
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    xs = src - mu_s
    xd = dst - mu_d
    Sigma = (xd.T @ xs) / src.shape[0]
    U, D, Vh = torch.linalg.svd(Sigma)
    S = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.det(U) * torch.det(Vh) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vh
    if with_scale:
        var_s = (xs ** 2).sum() / src.shape[0]
        s = (D * torch.diag(S)).sum() / var_s.clamp_min(1e-12)
    else:
        s = torch.tensor(1.0, device=src.device, dtype=src.dtype)
    t = mu_d - s * (R @ mu_s)
    return s, R, t


# --------------------------------------------------------------------------- #
# Camera helpers (OpenCV world->camera extrinsics)
# --------------------------------------------------------------------------- #
def look_at_extrinsics(eye, target, up=(0.0, 0.0, 1.0), device=None, dtype=torch.float32):
    """OpenCV world->camera extrinsic (4x4). +Z forward (toward target), +Y down."""
    eye = torch.as_tensor(eye, device=device, dtype=dtype)
    target = torch.as_tensor(target, device=device, dtype=dtype)
    up = torch.as_tensor(up, device=device, dtype=dtype)
    z = torch.nn.functional.normalize(target - eye, dim=0)      # forward
    x = torch.nn.functional.normalize(torch.cross(z, up, dim=0), dim=0)  # right
    y = torch.cross(z, x, dim=0)                                # down (=z x x)
    R = torch.stack([x, y, z], dim=0)                           # world->camera rotation (rows)
    t = -R @ eye
    E = torch.eye(4, device=device, dtype=dtype)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


def extrinsics_rotation(E: torch.Tensor) -> torch.Tensor:
    return E[:3, :3]


def camera_center(E: torch.Tensor) -> torch.Tensor:
    """World-space camera center from a world->camera extrinsic."""
    R = E[:3, :3]
    t = E[:3, 3]
    return -R.T @ t


def fov_to_intrinsics_normalized(fov_x_rad, fov_y_rad, device=None, dtype=torch.float32):
    """Normalized OpenCV intrinsics (fx,fy,cx,cy as image fractions), principal point centered."""
    fx = 0.5 / np.tan(0.5 * fov_x_rad)
    fy = 0.5 / np.tan(0.5 * fov_y_rad)
    K = torch.eye(3, device=device, dtype=dtype)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = 0.5
    K[1, 2] = 0.5
    return K


def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Normalized OpenCV intrinsics -> OpenGL projection (identical to AniGen MeshRenderer)."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


# --------------------------------------------------------------------------- #
# Skeleton: topological order + LBS
# --------------------------------------------------------------------------- #
def topological_order(parents: np.ndarray):
    """Return joint indices with every parent before its children (roots first)."""
    parents = np.asarray(parents).astype(np.int64)
    M = len(parents)
    children = {i: [] for i in range(M)}
    roots = []
    for j in range(M):
        p = int(parents[j])
        if p < 0 or p == j:
            roots.append(j)
        else:
            children[p].append(j)
    order = []
    stack = list(roots)
    seen = set()
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        order.append(j)
        stack.extend(children[j])
    # any leftover (cycles / unreachable) appended so the map stays total
    for j in range(M):
        if j not in seen:
            order.append(j)
    return order


class Skeleton:
    """Holds rest skeleton on a device and applies LBS given per-joint rotations."""

    def __init__(self, joints, parents, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.joints = torch.as_tensor(np.asarray(joints), device=device, dtype=dtype)  # [M,3]
        self.parents = np.asarray(parents).astype(np.int64)                            # [M]
        self.M = self.joints.shape[0]
        self.order = topological_order(self.parents)

    def forward_kinematics(self, R_local: torch.Tensor):
        """R_local [M,3,3] -> (GR [M,3,3], gp [M,3]) global rotations & posed pivots."""
        GR = [None] * self.M
        gp = [None] * self.M
        for j in self.order:
            p = int(self.parents[j])
            if p < 0 or p == j:
                GR[j] = R_local[j]
                gp[j] = self.joints[j]
            else:
                GR[j] = GR[p] @ R_local[j]
                gp[j] = gp[p] + GR[p] @ (self.joints[j] - self.joints[p])
        return torch.stack(GR, 0), torch.stack(gp, 0)

    def lbs(self, vertices: torch.Tensor, weights: torch.Tensor, R_local: torch.Tensor):
        """Deform vertices by LBS.

        vertices [V,3], weights [V,M], R_local [M,3,3] -> deformed [V,3] (canonical frame).
        """
        GR, gp = self.forward_kinematics(R_local)                 # [M,3,3], [M,3]
        offset = gp - torch.einsum("mij,mj->mi", GR, self.joints)  # [M,3]  = gp - GR p
        A = torch.einsum("vm,mij->vij", weights, GR)              # [V,3,3]
        b = weights @ offset                                     # [V,3]
        return torch.einsum("vij,vj->vi", A, vertices) + b


def apply_similarity(v: torch.Tensor, scale, Rg: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
    """v [V,3] -> scale * (Rg @ v^T)^T + tg."""
    return scale * (v @ Rg.T) + tg


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    dev = "cpu"

    # rot6d round-trip
    R = project_to_so3(torch.randn(3, 3))
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))
    assert torch.allclose(R, R2, atol=1e-5), "rot6d round-trip failed"

    # identity_rot6d -> I
    assert torch.allclose(rot6d_to_matrix(identity_rot6d(1))[0], torch.eye(3), atol=1e-6)

    # axis-angle: 90deg about z
    Rz = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.pi / 2]))
    exp = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert torch.allclose(Rz, exp, atol=1e-5), f"axis-angle z failed:\n{Rz}"

    # quat identity
    assert torch.allclose(quat_to_matrix(torch.tensor([1.0, 0, 0, 0])), torch.eye(3), atol=1e-6)

    # umeyama recovers a known similarity
    A = torch.randn(50, 3)
    s_gt = torch.tensor(2.3)
    R_gt = project_to_so3(torch.randn(3, 3))
    t_gt = torch.tensor([1.0, -2.0, 0.5])
    B = s_gt * (A @ R_gt.T) + t_gt
    s, Rr, t = umeyama(A, B)
    assert abs(s - s_gt) < 1e-3 and torch.allclose(Rr, R_gt, atol=1e-3) and torch.allclose(t, t_gt, atol=1e-3), \
        f"umeyama failed s={s} R err={(Rr-R_gt).abs().max()} t={t}"

    # camera round-trip: center recovered
    E = look_at_extrinsics([2.0, 0.5, 1.0], [0, 0, 0], up=[0, 0, 1])
    C = camera_center(E)
    assert torch.allclose(C, torch.tensor([2.0, 0.5, 1.0]), atol=1e-5), f"camera center {C}"
    # extrinsic rotation orthonormal, det +1
    Rc = extrinsics_rotation(E)
    assert torch.allclose(Rc @ Rc.T, torch.eye(3), atol=1e-5) and torch.det(Rc) > 0

    # LBS: identity rotations reproduce rest mesh
    joints = np.array([[0, 0, 0], [0, 0, 0.3], [0, 0.2, 0.3]], dtype=np.float32)
    parents = np.array([-1, 0, 1], dtype=np.int64)
    sk = Skeleton(joints, parents, device=dev)
    V = 20
    verts = torch.randn(V, 3) * 0.3
    w = torch.rand(V, 3)
    w = w / w.sum(1, keepdim=True)
    R_id = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)
    rest = sk.lbs(verts, w, R_id)
    assert torch.allclose(rest, verts, atol=1e-5), f"LBS identity failed max err {(rest-verts).abs().max()}"

    # LBS: rotating a single-root skeleton (all weight on root) = rigid rotation about root pivot
    Rroot = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.pi / 2]))
    R_local = torch.stack([Rroot, torch.eye(3), torch.eye(3)], 0)
    w_root = torch.zeros(V, 3)
    w_root[:, 0] = 1.0
    out = sk.lbs(verts, w_root, R_local)
    exp = (verts - sk.joints[0]) @ Rroot.T + sk.joints[0]
    assert torch.allclose(out, exp, atol=1e-5), f"LBS root-rot failed max err {(out-exp).abs().max()}"

    # FK: rotating root by 90deg about z moves child pivot correctly
    GR, gp = sk.forward_kinematics(R_local)
    # child 1 pivot at (0,0,0.3): rotating about z by 90 keeps it (on axis) -> unchanged
    assert torch.allclose(gp[1], torch.tensor([0.0, 0.0, 0.3]), atol=1e-5), f"gp1 {gp[1]}"
    # child 2 pivot at (0,0.2,0.3) -> rotate xy by 90 about z: (0,0.2)->(-0.2,0)
    assert torch.allclose(gp[2], torch.tensor([-0.2, 0.0, 0.3]), atol=1e-5), f"gp2 {gp[2]}"

    # similarity
    v = torch.randn(5, 3)
    out = apply_similarity(v, 2.0, torch.eye(3), torch.tensor([1.0, 0, 0]))
    assert torch.allclose(out, 2.0 * v + torch.tensor([1.0, 0, 0]), atol=1e-6)

    print("geom.py self-tests PASSED")


if __name__ == "__main__":
    _selftest()
