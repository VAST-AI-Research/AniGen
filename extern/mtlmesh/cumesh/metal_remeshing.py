"""
Remeshing via narrow-band dual contouring.
Uses MtlBVH for distance queries and Metal hash tables for spatial lookups.
"""
from typing import Tuple
import torch
from tqdm import tqdm
from . import _C


def _init_hashmap(resolution, capacity, device):
    VOL = resolution * resolution * resolution
    if VOL < 2**32:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"Spatial size too large: volume {VOL} > 2^64")
    hashmap_vals = torch.empty((capacity,), dtype=torch.uint32, device=device)
    return hashmap_keys, hashmap_vals


def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remesh using narrow-band UDF and dual contouring.

    Args:
        vertices: (N, 3) float32 tensor.
        faces: (M, 3) int32 tensor.
        center: (3,) center of the domain.
        scale: float size of the domain.
        resolution: int grid resolution.
        band: width of the narrow band in voxel units.
        project_back: float ratio to project vertices back to original mesh.
        verbose: print progress.
        bvh: prebuilt BVH (optional).

    Returns:
        (V_new, F_new): Tuple of new vertices and faces.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and vertices.dtype == torch.float32
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == torch.int32
    assert center.ndim == 1 and center.shape[0] == 3

    device = vertices.device

    # Topology tables (cached as function attributes)
    if not hasattr(remesh_narrow_band_dc, "edge_neighbor_voxel_offset"):
        remesh_narrow_band_dc.edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int32).unsqueeze(0).to(device)

    if not hasattr(remesh_narrow_band_dc, "quad_split_1_p"):
        remesh_narrow_band_dc.quad_split_1_n = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long).to(device)
        remesh_narrow_band_dc.quad_split_1_p = torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long).to(device)
        remesh_narrow_band_dc.quad_split_2_n = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long).to(device)
        remesh_narrow_band_dc.quad_split_2_p = torch.tensor([0, 3, 1, 3, 2, 1], dtype=torch.long).to(device)

    OFFSETS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=torch.int32, device=device)

    # Build BVH if needed
    if bvh is None:
        from mtlbvh import MtlBVH
        if verbose:
            print("Building BVH...")
        bvh = MtlBVH(vertices, faces)

    eps = band * scale / resolution

    # Octree-like sparse grid construction
    base_resolution = resolution
    while base_resolution > 32:
        assert base_resolution % 2 == 0
        base_resolution //= 2

    coords = torch.stack(torch.meshgrid(
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        indexing='ij'
    ), dim=-1).int().reshape(-1, 3)

    pbar = tqdm(total=int(torch.log2(torch.tensor(resolution // base_resolution)).item()) + 1,
                desc="Building Sparse Grid", disable=not verbose)

    while True:
        cell_size = scale / base_resolution
        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center
        distances = bvh.unsigned_distance(pts)[0]
        distances -= eps
        distances = torch.abs(distances)
        subdiv_mask = distances < 0.87 * cell_size
        coords = coords[subdiv_mask]

        if base_resolution >= resolution:
            break

        base_resolution *= 2
        coords *= 2
        coords = coords.unsqueeze(1) + OFFSETS.unsqueeze(0)
        coords = coords.reshape(-1, 3)
        coords = torch.unique(coords, dim=0)
        pbar.update(1)

    # Dual contouring
    Nvox = coords.shape[0]
    hashmap_vox = _init_hashmap(resolution, 2 * Nvox, device)
    _C.hashmap_insert_3d_idx_as_val(
        *hashmap_vox,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1),
        resolution, resolution, resolution
    )

    coords = coords.contiguous()
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(
        *hashmap_vox, coords, resolution, resolution, resolution
    )
    Nvert = grid_verts.shape[0]

    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    distances_vert = bvh.unsigned_distance(pts_vert)[0]
    distances_vert -= eps
    pbar.update(1)
    pbar.close()

    if verbose:
        print("Running Dual Contouring...")

    hashmap_vert = _init_hashmap(resolution + 1, 2 * Nvert, device)
    _C.hashmap_insert_3d_idx_as_val(
        *hashmap_vert,
        torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1),
        resolution + 1, resolution + 1, resolution + 1
    )

    dual_verts, intersected = _C.simple_dual_contour(
        *hashmap_vert, coords, distances_vert,
        resolution + 1, resolution + 1, resolution + 1
    )

    # Topology generation
    edge_neighbor_voxel = coords.reshape(Nvox, 1, 1, 3) + remesh_narrow_band_dc.edge_neighbor_voxel_offset
    connected_voxel = edge_neighbor_voxel[intersected != 0]
    intersected = intersected[intersected != 0]
    M = connected_voxel.shape[0]
    connected_voxel_hash_key = torch.cat([
        torch.zeros((M * 4, 1), dtype=torch.int, device=device),
        connected_voxel.reshape(-1, 3)
    ], dim=1)
    connected_voxel_indices = _C.hashmap_lookup_3d(
        *hashmap_vox, connected_voxel_hash_key,
        resolution, resolution, resolution
    ).reshape(M, 4).int()
    connected_voxel_valid = (connected_voxel_indices != 0xffffffff).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].int()
    intersected_dir = intersected[connected_voxel_valid].int()
    L = quad_indices.shape[0]

    # Remove unreferenced vertices
    unique_verts = torch.unique(quad_indices.reshape(-1))
    dual_verts = dual_verts[unique_verts]
    vert_map = torch.zeros((Nvox,), dtype=torch.int32, device=device)
    vert_map[unique_verts] = torch.arange(unique_verts.shape[0], dtype=torch.int32, device=device)
    quad_indices = vert_map[quad_indices]

    # Construct triangles
    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center
    atempt_triangles_0 = torch.where(
        (intersected_dir == 1).unsqueeze(1),
        quad_indices[:, remesh_narrow_band_dc.quad_split_1_p],
        quad_indices[:, remesh_narrow_band_dc.quad_split_1_n]
    )
    normals0 = torch.cross(
        mesh_vertices[atempt_triangles_0[:, 1]] - mesh_vertices[atempt_triangles_0[:, 0]],
        mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 0]]
    )
    normals1 = torch.cross(
        mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 1]],
        mesh_vertices[atempt_triangles_0[:, 3]] - mesh_vertices[atempt_triangles_0[:, 1]]
    )
    align0 = (normals0 * normals1).sum(dim=1).abs()

    atempt_triangles_1 = torch.where(
        (intersected_dir == 1).unsqueeze(1),
        quad_indices[:, remesh_narrow_band_dc.quad_split_2_p],
        quad_indices[:, remesh_narrow_band_dc.quad_split_2_n]
    )
    normals0 = torch.cross(
        mesh_vertices[atempt_triangles_1[:, 1]] - mesh_vertices[atempt_triangles_1[:, 0]],
        mesh_vertices[atempt_triangles_1[:, 2]] - mesh_vertices[atempt_triangles_1[:, 0]]
    )
    normals1 = torch.cross(
        mesh_vertices[atempt_triangles_1[:, 2]] - mesh_vertices[atempt_triangles_1[:, 1]],
        mesh_vertices[atempt_triangles_1[:, 3]] - mesh_vertices[atempt_triangles_1[:, 1]]
    )
    align1 = (normals0 * normals1).sum(dim=1).abs()

    mesh_triangles = torch.where(
        (align0 > align1).unsqueeze(1), atempt_triangles_0, atempt_triangles_1
    ).reshape(-1, 3)

    # Optional: project back
    if project_back > 0:
        if verbose:
            print("Projecting back to original mesh...")
        _, face_id, uvw = bvh.unsigned_distance(mesh_vertices, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        mesh_vertices -= project_back * (mesh_vertices - projected_verts)

    return mesh_vertices, mesh_triangles.int()
