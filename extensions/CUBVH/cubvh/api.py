import contextlib
import numpy as np
import torch

# CUDA extension
import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

class cuBVH:
    def __init__(self, vertices=None, triangles=None, *, state=None, device=None):
        if state is None and (vertices is None or triangles is None):
            raise ValueError("cuBVH requires either vertices/triangles or a serialized state.")

        self._impl = None
        self._impl_device = None
        self._state = None
        self._exhaustive = None
        self._is_exhaustive = False
        self.device = torch.device('cpu')

        target_device = self._parse_device(device)

        if state is not None:
            self._load_state(state)
        else:
            self._build_from_mesh(vertices, triangles, target_device)

        try:
            self.to(target_device)
        except RuntimeError:
            if target_device.type != 'cpu':
                self.to(torch.device('cpu'))
            else:
                raise

    @staticmethod
    def _parse_device(device):
        if device is None:
            return torch.device('cpu')
        return torch.device(device)

    def _build_from_mesh(self, vertices, triangles, build_device):
        if torch.is_tensor(vertices):
            vertices_arr = vertices.detach().cpu().numpy()
        else:
            vertices_arr = np.asarray(vertices, dtype=np.float32)

        if torch.is_tensor(triangles):
            triangles_arr = triangles.detach().cpu().numpy()
        else:
            triangles_arr = np.asarray(triangles, dtype=np.int32)

        vertices_arr = np.asarray(vertices_arr, dtype=np.float32)
        triangles_arr = np.asarray(triangles_arr, dtype=np.int32)

        if triangles_arr.shape[0] == 0:
            raise ValueError("cuBVH requires at least one triangle.")

        if triangles_arr.shape[0] < 8:
            self._build_exhaustive(vertices_arr, triangles_arr)
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to build a cuBVH from mesh data.")

        if build_device.type == 'cuda':
            build_cuda = build_device
        else:
            index = torch.cuda.current_device()
            build_cuda = torch.device('cuda', index)

        with self._cuda_device_guard(build_cuda):
            impl = _backend.create_cuBVH(vertices_arr, triangles_arr)
            tri_pos, tri_ids, node_mins, node_maxs, node_children = impl.export_state()

        self._state = {
            "mode": torch.tensor(0, dtype=torch.int32),
            "triangles": tri_pos.detach().clone().cpu().contiguous(),
            "triangle_ids": tri_ids.detach().clone().cpu().contiguous(),
            "node_mins": node_mins.detach().clone().cpu().contiguous(),
            "node_maxs": node_maxs.detach().clone().cpu().contiguous(),
            "node_children": node_children.detach().clone().cpu().contiguous(),
        }

        self._impl = impl
        self._impl_device = build_cuda
        self._is_exhaustive = False
        self.device = build_cuda

    def _build_exhaustive(self, vertices_arr, triangles_arr):
        vertices_tensor = torch.as_tensor(vertices_arr, dtype=torch.float32).contiguous().clone()
        faces_tensor = torch.as_tensor(triangles_arr, dtype=torch.long).contiguous().clone()
        tri_positions = torch.as_tensor(vertices_arr[triangles_arr], dtype=torch.float32).contiguous().clone()
        triangle_ids = torch.arange(triangles_arr.shape[0], dtype=torch.long).contiguous()
        node_mins = torch.zeros((0, 3), dtype=torch.float32)
        node_maxs = torch.zeros((0, 3), dtype=torch.float32)
        node_children = torch.zeros((0, 2), dtype=torch.int32)

        self._state = {
            "mode": torch.tensor(1, dtype=torch.int32),
            "vertices": vertices_tensor,
            "faces": faces_tensor,
            "triangles": tri_positions,
            "triangle_ids": triangle_ids,
            "node_mins": node_mins,
            "node_maxs": node_maxs,
            "node_children": node_children,
        }

        self._exhaustive = ExaustiveSearcher(vertices_tensor, faces_tensor, torch.device('cpu'))
        self._impl = None
        self._impl_device = None
        self._is_exhaustive = True
        self.device = torch.device('cpu')

    def _pull_state_from_impl(self):
        if self._is_exhaustive:
            return {key: value.clone() for key, value in self._state.items()}

        with self._cuda_device_guard(self._impl_device):
            tri_pos, tri_ids, node_mins, node_maxs, node_children = self._impl.export_state()
        return {
            "mode": torch.tensor(0, dtype=torch.int32),
            "triangles": tri_pos.detach().clone().cpu().contiguous(),
            "triangle_ids": tri_ids.detach().clone().cpu().contiguous(),
            "node_mins": node_mins.detach().clone().cpu().contiguous(),
            "node_maxs": node_maxs.detach().clone().cpu().contiguous(),
            "node_children": node_children.detach().clone().cpu().contiguous(),
        }

    def _load_state(self, state):
        mode_value = state.get("mode", None)
        if mode_value is None:
            mode_tensor = torch.tensor(0, dtype=torch.int32)
        else:
            mode_tensor = torch.as_tensor(mode_value, dtype=torch.int32, device='cpu').contiguous().clone()

        mode = int(mode_tensor.item())

        if mode == 1:
            required = {"vertices", "faces"}
            missing = required.difference(state.keys())
            if missing:
                raise ValueError(f"Serialized state is missing keys for exhaustive mode: {sorted(missing)}")

            vertices = torch.as_tensor(state["vertices"], dtype=torch.float32, device='cpu').contiguous().clone()
            faces = torch.as_tensor(state["faces"], dtype=torch.long, device='cpu').contiguous().clone()
            triangles = torch.as_tensor(
                state.get("triangles", vertices[faces]),
                dtype=torch.float32,
                device='cpu',
            ).contiguous().clone()
            triangle_ids = torch.as_tensor(
                state.get("triangle_ids", torch.arange(faces.shape[0], dtype=torch.long)),
                dtype=torch.long,
                device='cpu',
            ).contiguous().clone()
            node_mins = torch.as_tensor(
                state.get("node_mins", torch.empty((0, 3), dtype=torch.float32)),
                dtype=torch.float32,
                device='cpu',
            ).contiguous().clone()
            node_maxs = torch.as_tensor(
                state.get("node_maxs", torch.empty((0, 3), dtype=torch.float32)),
                dtype=torch.float32,
                device='cpu',
            ).contiguous().clone()
            node_children = torch.as_tensor(
                state.get("node_children", torch.empty((0, 2), dtype=torch.int32)),
                dtype=torch.int32,
                device='cpu',
            ).contiguous().clone()

            self._state = {
                "mode": mode_tensor,
                "vertices": vertices,
                "faces": faces,
                "triangles": triangles,
                "triangle_ids": triangle_ids,
                "node_mins": node_mins,
                "node_maxs": node_maxs,
                "node_children": node_children,
            }

            self._impl = None
            self._impl_device = None
            self._is_exhaustive = True
            self._exhaustive = ExaustiveSearcher(vertices, faces, torch.device('cpu'))
            self.device = torch.device('cpu')
            return

        required = {"triangles", "triangle_ids", "node_mins", "node_maxs", "node_children"}
        missing = required.difference(state.keys())
        if missing:
            raise ValueError(f"Serialized state is missing keys: {sorted(missing)}")

        self._state = {
            "mode": mode_tensor,
            "triangles": torch.as_tensor(state["triangles"], dtype=torch.float32, device='cpu').contiguous().clone(),
            "triangle_ids": torch.as_tensor(state["triangle_ids"], dtype=torch.long, device='cpu').contiguous().clone(),
            "node_mins": torch.as_tensor(state["node_mins"], dtype=torch.float32, device='cpu').contiguous().clone(),
            "node_maxs": torch.as_tensor(state["node_maxs"], dtype=torch.float32, device='cpu').contiguous().clone(),
            "node_children": torch.as_tensor(state["node_children"], dtype=torch.int32, device='cpu').contiguous().clone(),
        }

        self._impl = None
        self._impl_device = None
        self._is_exhaustive = False
        self._exhaustive = None
        self.device = torch.device('cpu')

    def _release_impl(self):
        if self._impl is None:
            return

        if self._impl_device is not None and self._impl_device.type == 'cuda':
            with self._cuda_device_guard(self._impl_device):
                torch.cuda.synchronize()

        self._impl = None
        self._impl_device = None

    def _instantiate_impl(self, cuda_device):
        triangles = self._state["triangles"].contiguous()
        triangle_ids = self._state["triangle_ids"].contiguous()
        node_mins = self._state["node_mins"].contiguous()
        node_maxs = self._state["node_maxs"].contiguous()
        node_children = self._state["node_children"].contiguous()

        with self._cuda_device_guard(cuda_device):
            self._impl = _backend.create_cuBVH_from_state(
                triangles,
                triangle_ids,
                node_mins,
                node_maxs,
                node_children,
            )

        self._impl_device = cuda_device
        self.device = cuda_device

    def to(self, device, *args, **kwargs):
        device = self._parse_device(device)

        if self._is_exhaustive:
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for cuBVH.to().")
            if device.type not in ('cpu', 'cuda'):
                raise ValueError(f"Unsupported device for cuBVH: {device}")

            vertices = self._state["vertices"].detach()
            faces = self._state["faces"].detach()
            if self._exhaustive is None:
                self._exhaustive = ExaustiveSearcher(vertices, faces, device)
            else:
                self._exhaustive = self._exhaustive.to(device)
            self.device = device
            return self

        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for cuBVH.to().")
            index = device.index if device.index is not None else torch.cuda.current_device()
            cuda_device = torch.device('cuda', index)
            if self._impl is not None and self._impl_device == cuda_device:
                self.device = cuda_device
                return self
            self._release_impl()
            self._instantiate_impl(cuda_device)
        elif device.type == 'cpu':
            self._release_impl()
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"Unsupported device for cuBVH: {device}")

        return self

    def _require_impl(self):
        if self._is_exhaustive:
            raise RuntimeError("cuBVH was built with fewer than 8 triangles; only unsigned_distance is supported.")
        if self._impl is None:
            raise RuntimeError("cuBVH is on CPU; call cuBVH.to('cuda') before querying.")
        return self._impl

    def ray_trace(self, rays_o, rays_d):
        impl = self._require_impl()
        if self.device.type != 'cuda':
            raise RuntimeError("cuBVH must be on a CUDA device to trace rays.")

        target_device = self.device
        rays_o = rays_o.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()
        rays_d = rays_d.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        n_rays = rays_o.shape[0]

        with self._cuda_device_guard(target_device):
            positions = torch.empty(n_rays, 3, dtype=torch.float32, device=target_device)
            face_id = torch.empty(n_rays, dtype=torch.int64, device=target_device)
            depth = torch.empty(n_rays, dtype=torch.float32, device=target_device)

            impl.ray_trace(rays_o, rays_d, positions, face_id, depth)

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        target_device = self.device
        positions = positions.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        n_points = positions.shape[0]

        if self._is_exhaustive:
            if self._exhaustive is None:
                raise RuntimeError("Exhaustive searcher not initialized.")

            distances, face_id, uvw = self._exhaustive.unsigned_distance(positions, return_uvw)
        else:
            impl = self._require_impl()
            if self.device.type != 'cuda':
                raise RuntimeError("cuBVH must be on a CUDA device to query distances.")

            with self._cuda_device_guard(target_device):
                distances = torch.empty(n_points, dtype=torch.float32, device=target_device)
                face_id = torch.empty(n_points, dtype=torch.int64, device=target_device)

                if return_uvw:
                    uvw = torch.empty(n_points, 3, dtype=torch.float32, device=target_device)
                else:
                    uvw = None

                impl.unsigned_distance(positions, distances, face_id, uvw)

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        impl = self._require_impl()
        if self.device.type != 'cuda':
            raise RuntimeError("cuBVH must be on a CUDA device to query distances.")

        target_device = self.device
        positions = positions.to(device=target_device, dtype=torch.float32, non_blocking=True).contiguous()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        n_points = positions.shape[0]

        with self._cuda_device_guard(target_device):
            distances = torch.empty(n_points, dtype=torch.float32, device=target_device)
            face_id = torch.empty(n_points, dtype=torch.int64, device=target_device)

            if return_uvw:
                uvw = torch.empty(n_points, 3, dtype=torch.float32, device=target_device)
            else:
                uvw = None

            impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode])

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    def export_state(self):
        return {key: value.clone() for key, value in self._state.items()}

    @property
    def triangles_cpu(self):
        if self._is_exhaustive:
            vertices = self._state["vertices"].detach().cpu()
            faces = self._state["faces"].detach().cpu().long()
            return vertices[faces].contiguous().clone().numpy()
        return self._state["triangles"].detach().cpu().clone().numpy()

    @property
    def bvh_nodes_cpu(self):
        if self._is_exhaustive:
            raise RuntimeError("Exhaustive search mode does not have BVH nodes.")
        return {
            "mins": self._state["node_mins"].detach().cpu().clone().numpy(),
            "maxs": self._state["node_maxs"].detach().cpu().clone().numpy(),
            "children": self._state["node_children"].detach().cpu().clone().numpy(),
        }

    def __getstate__(self):
        return {
            "state": self.export_state(),
            "device": "cpu",
        }

    def __setstate__(self, state):
        serialized_state = state["state"]
        load_device = state.get("device", "cpu")
        if isinstance(load_device, torch.device):
            if load_device.type == "cuda":
                load_device = "cpu"
        elif isinstance(load_device, str) and load_device.startswith("cuda"):
            load_device = "cpu"
        self.__init__(state=serialized_state, device=load_device)

    @contextlib.contextmanager
    def _cuda_device_guard(self, device):
        if device is None or device.type != 'cuda':
            yield
        else:
            with torch.cuda.device(device):
                yield

def floodfill(grid):
    # grid: torch.Tensor, uint8, [B, H, W, D] or [H, W, D]
    # return: torch.Tensor, int32, [B, H, W, D] or [H, W, D], label of the connected component (value can be 0 to H*W*D-1, not remapped!)

    grid = grid.contiguous()
    if not grid.is_cuda: grid = grid.cuda()

    if grid.dim() == 3:
        mask = _backend.floodfill(grid.unsqueeze(0)).squeeze(0)
    else:
        mask = _backend.floodfill(grid)

    return mask


class cuHashTable:
    """
    Python wrapper around the CUDA ND integer hash table.

    - Default dimensionality is 3; can be changed via num_dims argument or set_num_dims.
    - Static table: prefer a single build() call; repeated insert() calls overwrite indices.
    """

    def __init__(self, num_dims: int = 3):
        # create implementation via factory (mirrors cuBVH style)
        self.impl = _backend.create_cuHashTable()
        self.impl.set_num_dims(int(num_dims))

    @property
    def num_dims(self) -> int:
        return int(self.impl.get_num_dims())

    def build(self, coords):
        """Build table from coordinates: coords [N,D] int32/cuda.
        Auto-sets capacity to max(16, 2*N)."""
        if coords.shape[1] != self.num_dims:
            self.impl.set_num_dims(int(coords.size(1)))
        self.impl.build(coords)

    def search(self, queries) -> torch.Tensor:
        """Search queries [M,D] -> indices [M] int32 on CUDA; -1 if not found."""
        assert queries.shape[1] == self.num_dims, f"queries must be {self.num_dims}D"
        return self.impl.search(queries)

    
def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):
    # coords: torch.Tensor, int32, [N, 3]
    # corners: torch.Tensor, float32, [N, 8]
    # iso: float
    # ensure_consistency: bool, whether to ensure shared corner values are consistent

    coords = coords.int().contiguous()
    corners = corners.float().contiguous()

    if not coords.is_cuda: coords = coords.cuda()
    if not corners.is_cuda: corners = corners.cuda()

    verts, tris = _backend.sparse_marching_cubes(coords, corners, iso, ensure_consistency)

    return verts, tris

# CPU hole filling numpy API
def fill_holes(vertices: np.ndarray, faces: np.ndarray, return_added: bool = False, check_containment: bool = True, eps: float = 1e-7, verbose: bool = False) -> np.ndarray:
    """
    Fill small holes in a triangular mesh using a CPU ear-clipping strategy.

    Args:
        vertices (np.ndarray float32 [N,3])
        faces (np.ndarray int32 [M,3])
        return_added: if True, return only newly added triangles; else full face list
        check_containment: avoid creating triangles containing other boundary verts
        eps: numeric epsilon
        verbose: print detailed logs from C++
    Returns:
        np.ndarray int32 [...,3]
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    faces = _backend.fill_holes(vertices, faces, return_added, check_containment, float(eps), bool(verbose))
    return np.asarray(faces, dtype=np.int32)

def merge_vertices(vertices: np.ndarray, faces: np.ndarray, threshold: float = 1e-3):
    """Merge vertices closer than threshold.
    Args:
        vertices (np.ndarray float32 [N,3])
        faces (np.ndarray int32 [M,3])
        threshold (float): distance threshold
    Returns:
        (vertices, faces) after merging
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    assert vertices.ndim==2 and vertices.shape[1]==3
    assert faces.ndim==2 and faces.shape[1]==3
    v_new, f_new = _backend.merge_vertices(vertices, faces, float(threshold))
    return np.asarray(v_new, dtype=np.float32), np.asarray(f_new, dtype=np.int32)


class HashTable:
    """
    CPU ND integer hash table (static, open-addressed). Mirrors cuHashTable but on host.
    """
    def __init__(self, num_dims: int = 3):
        # constructed directly from backend class
        self.impl = _backend.HashTable()
        self.impl.set_num_dims(int(num_dims))

    @property
    def num_dims(self) -> int:
        return int(self.impl.get_num_dims())

    def build(self, coords):
        """Build table from coordinates: coords [N,D] int32/CPU.
        Auto-sets capacity to max(16, 2*N)."""
        if coords.shape[1] != self.num_dims:
            self.impl.set_num_dims(int(coords.shape[1]))
        coords = coords.int().contiguous().cpu()
        self.impl.build(coords)

    def search(self, queries):
        """Search queries [M,D] -> indices [M] int32/CPU; -1 if not found."""
        assert queries.shape[1] == self.num_dims, f"queries must be {self.num_dims}D"
        queries = queries.int().contiguous().cpu()
        return self.impl.search(queries)

def sparse_marching_cubes_cpu(coords, corners, iso: float, ensure_consistency: bool = False):
    """CPU sparse marching cubes wrapper.
    Args:
        coords: (N,3) int32 voxel coordinates (torch.Tensor or np.ndarray)
        corners: (N,8) float32 corner SDF values (torch.Tensor or np.ndarray)
        iso: isovalue
        ensure_consistency: average shared corners across voxels before extraction
    Returns:
        (vertices, faces): np.ndarray float32 [M,3], np.ndarray int32 [T,3]
    """
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    if torch.is_tensor(corners):
        corners = corners.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=np.int32)
    corners = np.asarray(corners, dtype=np.float32)
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be [N,3]"
    assert corners.ndim == 2 and corners.shape[1] == 8, "corners must be [N,8]"
    v, f = _backend.sparse_marching_cubes_cpu(coords, corners, float(iso), bool(ensure_consistency))
    return np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32)


def decimate(vertices: np.ndarray, faces: np.ndarray, target_vertices: int):
    """CPU quadric-error simplification to target number of vertices.
    Args:
        vertices: np.ndarray float32 or float64 [N,3]
        faces: np.ndarray int32 [M,3]
        target_vertices: desired vertex count after decimation
    Returns:
        (vertices, faces): simplified mesh
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    faces = faces.astype(np.int32)
    v, f = _backend.decimate(vertices, faces, int(target_vertices))
    return v, f


def parallel_decimate(vertices: np.ndarray, faces: np.ndarray, target_vertices: int):
    """CPU batch-parallel decimation to target number of vertices.
    Args:
        vertices: np.ndarray float32 or float64 [N,3]
        faces: np.ndarray int32 [M,3]
        target_vertices: desired vertex count after decimation
    Returns:
        (vertices, faces): simplified mesh
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    faces = faces.astype(np.int32)
    v, f = _backend.parallel_decimate(vertices, faces, int(target_vertices))
    return v, f


class ExaustiveSearcher:
    """Fallback distance queries via exhaustive point-triangle search."""
    def __init__(self, vertices, triangles, device):
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)

        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for ExaustiveSearcher.")

        vertices_tensor = torch.as_tensor(vertices, dtype=torch.float32)
        faces_tensor = torch.as_tensor(triangles, dtype=torch.long)

        if faces_tensor.numel() == 0:
            raise ValueError("ExaustiveSearcher requires at least one triangle.")

        self.device = device
        self.vertices = vertices_tensor.to(device=device, dtype=torch.float32).contiguous().clone()
        self.faces = faces_tensor.to(device=device, dtype=torch.long).contiguous().clone()

        self._refresh_triangle_vertices()

    def _refresh_triangle_vertices(self):
        if self.faces.numel() == 0:
            self._triangle_vertices = self.vertices.new_zeros((0, 3, 3))
        else:
            self._triangle_vertices = self.vertices[self.faces]

    def to(self, device):
        device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for ExaustiveSearcher.to().")
        if device == self.device:
            return self

        self.vertices = self.vertices.to(device=device, non_blocking=True)
        self.faces = self.faces.to(device=device)
        self.device = device
        self._refresh_triangle_vertices()
        return self

    def unsigned_distance(self, positions, return_uvw=False):
        if self._triangle_vertices.numel() == 0:
            raise RuntimeError("ExaustiveSearcher requires triangles to compute distances.")

        points = positions.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
        n_points = points.shape[0]

        best_sqdist = torch.full((n_points,), float('inf'), dtype=torch.float32, device=self.device)
        best_face = torch.full((n_points,), -1, dtype=torch.int64, device=self.device)
        best_bary = torch.zeros((n_points, 3), dtype=torch.float32, device=self.device) if return_uvw else None

        for idx in range(self._triangle_vertices.shape[0]):
            tri = self._triangle_vertices[idx]
            sq_dist, bary = self._point_triangle_distance(points, tri)
            better = sq_dist < best_sqdist
            if better.any():
                best_sqdist[better] = sq_dist[better]
                best_face[better] = idx
                if return_uvw:
                    best_bary[better] = bary[better]

        distances = torch.sqrt(best_sqdist.clamp_min(0.0))
        if return_uvw:
            return distances, best_face, best_bary
        return distances, best_face, None

    @staticmethod
    def _point_triangle_distance(points, tri):
        eps = 1e-12

        a, b, c = tri[0], tri[1], tri[2]
        ab = b - a
        ac = c - a
        ap = points - a
        d1 = (ab * ap).sum(dim=-1)
        d2 = (ac * ap).sum(dim=-1)

        bp = points - b
        d3 = (ab * bp).sum(dim=-1)
        d4 = (ac * bp).sum(dim=-1)

        cp = points - c
        d5 = (ab * cp).sum(dim=-1)
        d6 = (ac * cp).sum(dim=-1)

        bary = torch.zeros(points.shape[0], 3, dtype=points.dtype, device=points.device)
        assigned = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)

        mask = (d1 <= 0) & (d2 <= 0)
        if mask.any():
            bary[mask, 0] = 1
            assigned |= mask

        mask = (d3 >= 0) & (d4 <= d3) & (~assigned)
        if mask.any():
            bary[mask, 1] = 1
            assigned |= mask

        mask = (d6 >= 0) & (d5 <= d6) & (~assigned)
        if mask.any():
            bary[mask, 2] = 1
            assigned |= mask

        vc = d1 * d4 - d3 * d2
        mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & (~assigned)
        if mask.any():
            denom = (d1 - d3)[mask]
            v = d1[mask] / (denom + eps)
            bary[mask, 0] = 1 - v
            bary[mask, 1] = v
            assigned |= mask

        vb = d5 * d2 - d1 * d6
        mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & (~assigned)
        if mask.any():
            denom = (d2 - d6)[mask]
            w = d2[mask] / (denom + eps)
            bary[mask, 0] = 1 - w
            bary[mask, 2] = w
            assigned |= mask

        va = d3 * d6 - d5 * d4
        mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & (~assigned)
        if mask.any():
            numerator = (d4 - d3)[mask]
            denom = (d4 - d3 + d5 - d6)[mask]
            w = numerator / (denom + eps)
            bary[mask, 1] = 1 - w
            bary[mask, 2] = w
            assigned |= mask

        mask = ~assigned
        if mask.any():
            denom = (va + vb + vc)[mask]
            v = vb[mask] / (denom + eps)
            w = vc[mask] / (denom + eps)
            bary[mask, 0] = 1 - v - w
            bary[mask, 1] = v
            bary[mask, 2] = w
            assigned |= mask

        if not assigned.all():
            remaining = ~assigned
            if remaining.any():
                verts = torch.stack([a, b, c], dim=0)
                diff = points[remaining].unsqueeze(1) - verts.unsqueeze(0)
                sq = (diff * diff).sum(dim=-1)
                _, min_idx = sq.min(dim=-1)
                bary_fallback = torch.zeros((min_idx.shape[0], 3), dtype=points.dtype, device=points.device)
                bary_fallback.scatter_(1, min_idx.unsqueeze(1), 1.0)
                bary[remaining] = bary_fallback
                assigned[remaining] = True

        closest = (
            bary[:, 0:1] * a.unsqueeze(0)
            + bary[:, 1:2] * b.unsqueeze(0)
            + bary[:, 2:3] * c.unsqueeze(0)
        )
        diff = closest - points
        sq_dist = (diff * diff).sum(dim=-1)

        return sq_dist, bary
