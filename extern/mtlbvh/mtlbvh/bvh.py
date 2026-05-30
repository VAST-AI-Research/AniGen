"""Python wrapper for MtlBVH — Metal-accelerated BVH queries."""
import torch
from . import _C

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

# Scalar output-dtype codes — matches MtlBVHImpl::OutDtype.
_OUT_DTYPE_IDS = {
    None: 0,
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

def _resolve_out_dtype(output_dtype):
    if output_dtype not in _OUT_DTYPE_IDS:
        raise ValueError(
            f"output_dtype must be None, torch.float32, torch.float16, or torch.bfloat16; "
            f"got {output_dtype}"
        )
    return _OUT_DTYPE_IDS[output_dtype]

class MtlBVH:
    """Metal-accelerated BVH for distance queries and ray tracing.

    Drop-in replacement for GPU BVH implementations.
    """
    def __init__(self, vertices, triangles):
        """Build BVH from mesh.

        Args:
            vertices: [N, 3] float32 (numpy or torch)
            triangles: [M, 3] int32 (numpy or torch)
        """
        # Normalize to torch tensors on CPU. The C++ constructor already pulls
        # inputs to CPU for the CPU-side BVH build; accept numpy only when
        # NumPy is available (some PyTorch builds compile without it).
        if not torch.is_tensor(vertices):
            if _HAS_NUMPY:
                vertices = torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32))
            else:
                vertices = torch.tensor(vertices, dtype=torch.float32)
        if not torch.is_tensor(triangles):
            if _HAS_NUMPY:
                triangles = torch.from_numpy(np.ascontiguousarray(triangles, dtype=np.int32))
            else:
                triangles = torch.tensor(triangles, dtype=torch.int32)

        vertices = vertices.detach().to(torch.float32).contiguous()
        triangles = triangles.detach().to(torch.int32).contiguous()

        assert triangles.shape[0] > 8, f"BVH needs at least 8 triangles, got {triangles.shape[0]}"

        self._impl = _C.MtlBVH(vertices, triangles)

    def unsigned_distance(self, positions, return_uvw=False, output_dtype=None):
        """Compute unsigned distance to mesh surface.

        Args:
            positions: [N, 3] torch tensor. Coerced to fp32 internally — the
                BVH traversal always runs in fp32.
            return_uvw: if True, also return barycentric coordinates.
            output_dtype: scalar dtype for the ``distances`` output. One of
                ``None`` / ``torch.float32`` / ``torch.float16`` /
                ``torch.bfloat16``. ``None`` defaults to fp32 (backward-compat).
                Only the scalar ``distances`` output is narrowed; ``uvw`` (vec3)
                always stays fp32.

        Returns:
            (distances [N] in `output_dtype`, face_id [N] int64, uvw [N,3] fp32 or None)
        """
        positions = positions.float().contiguous()
        out_id = _resolve_out_dtype(output_dtype)
        return self._impl.unsigned_distance(positions, return_uvw, out_id)

    def signed_distance(self, positions, return_uvw=False, mode='watertight', output_dtype=None):
        """Compute signed distance to mesh surface.

        Args:
            positions: [N, 3] torch tensor (coerced to fp32 internally).
            return_uvw: if True, also return barycentric coordinates.
            mode: 'watertight' or 'raystab'.
            output_dtype: scalar dtype for the ``distances`` output. See
                ``unsigned_distance`` — only the scalar ``distances`` output is
                narrowed; ``uvw`` (vec3) always stays fp32.

        Returns:
            (distances [N] in `output_dtype`, face_id [N] int64, uvw [N,3] fp32 or None)
        """
        positions = positions.float().contiguous()
        out_id = _resolve_out_dtype(output_dtype)
        return self._impl.signed_distance(positions, return_uvw, _sdf_mode_to_id[mode], out_id)

    def ray_trace(self, rays_o, rays_d, output_dtype=None):
        """Ray-triangle intersection.

        Args:
            rays_o: [N, 3] torch tensor (ray origins, coerced to fp32).
            rays_d: [N, 3] torch tensor (ray directions, coerced to fp32).
            output_dtype: scalar dtype for the ``depth`` output. See
                ``unsigned_distance`` — only the scalar ``depth`` output is
                narrowed; hit ``positions`` (vec3) always stay fp32.

        Returns:
            (positions [N,3] fp32, face_id [N] int64, depth [N] in `output_dtype`)
        """
        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()
        out_id = _resolve_out_dtype(output_dtype)
        return self._impl.ray_trace(rays_o, rays_d, out_id)
