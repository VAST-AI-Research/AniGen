import torch

# CUDA extension
from . import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

class cuBVH():
    def __init__(self, vertices, triangles):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        # check inputs
        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        self.impl = _backend.create_cuBVH(vertices, triangles)

    def ray_trace(self, rays_o, rays_d):
        # rays_o: torch.Tensor, float, [N, 3]
        # rays_d: torch.Tensor, float, [N, 3]

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        # init output buffers
        positions = torch.empty(N, 3, dtype=torch.float32, device=rays_o.device)
        face_id = torch.empty(N, dtype=torch.int64, device=rays_o.device)
        depth = torch.empty(N, dtype=torch.float32, device=rays_o.device)
        
        self.impl.ray_trace(rays_o, rays_d, positions, face_id, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        self.impl.unsigned_distance(positions, distances, face_id, uvw) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw


    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        self.impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode]) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw
