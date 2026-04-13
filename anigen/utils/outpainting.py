import torch
import numpy as np

def voxels_to_mesh(voxels):
    """
    Convert voxel integer coordinates into a triangle mesh.

    Args:
        voxels (torch.Tensor): Tensor of shape [N, 3] indicating voxel integer coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Vertices of shape [8*N, 3] and faces of shape [6*2*N, 3].
    """
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.cpu().numpy()
    cube_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ])
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [1, 2, 6], [1, 6, 5],  # Right face
        [0, 3, 7], [0, 7, 4],  # Left face
    ])
    N = voxels.shape[0]
    voxel_vertices = (voxels[:, None, :] + cube_vertices[None, :, :]).reshape(-1, 3)
    voxel_faces = (np.arange(N)[:, None, None] * 8 + cube_faces[None, :, :]).reshape(-1, 3)
    return voxel_vertices, voxel_faces

