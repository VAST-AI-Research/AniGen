from typing import *
import gc
import numpy as np
import torch
import utils3d
import torch as _torch
if _torch.cuda.is_available():
    import nvdiffrast.torch as dr
else:
    import mtldiffrast.torch as dr   # API-compatible Metal replacement
# Active compute device. The original code hardcodes device=_DEV in factory calls
# (torch.tensor(..., device=_DEV)), which raises on a non-CUDA build; route those to
# MPS/CPU on Apple Silicon. (.cuda()/.to('cuda') are remapped globally by anigen_mps.)
_DEV = 'cuda' if _torch.cuda.is_available() else ('mps' if _torch.backends.mps.is_available() else 'cpu')
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..renderers import GaussianRenderer
from ..representations import Strivec, Gaussian, MeshExtractResult


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rast_backend():
    # utils3d.torch.RastContext only accepts {'gl', 'cuda'}; it has no 'cpu' backend.
    # On Apple Silicon, anigen_mps.install_nvdiffrast_alias() maps the nvdiffrast
    # context classes (RasterizeCudaContext/RasterizeGLContext) to mtldiffrast's
    # MtlRasterizeContext, so 'cuda' routes through Metal. Always request 'cuda'.
    return 'cuda'


@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend=_rast_backend())
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        view = views[i]
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None], faces, resolution, resolution, view=view, projection=projection
        )
        face_id = buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views
    
    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[connected_components[i]] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)
    
    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces')
    if inner_face_indices.shape[0] == 0:
        return verts, faces
    
    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es['weight'] = dual_edges_weights.cpu().numpy()
    
    ### source and target
    g.add_vertex('s')
    g.add_vertex('t')
    
    ### connect invisible faces to source
    g.add_edges([(f, 's') for f in inner_face_indices], attributes={'weight': torch.ones(inner_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
    
    ### connect outer faces to target
    g.add_edges([(f, 't') for f in outer_face_indices], attributes={'weight': torch.ones(outer_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
                
    ### solve mincut
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')
    
    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue
        
        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue
            
        valid_remove_cc.append(cc)
        
    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    
    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')
            
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(verts, device=_DEV, dtype=torch.float32), torch.tensor(faces, device=_DEV, dtype=torch.int32)

    return verts, faces


def remove_sliver_faces(vertices: np.array, faces: np.array, edge_mult: float = 10.0):
    """Drop degenerate sliver triangles whose longest edge is a gross outlier.

    FlexiCubes' learned-deformation interpolation (``flexicubes._linear_interp``)
    can extrapolate a dual vertex far outside its cell when the alpha-scaled SDF
    endpoints are near-equal (denominator -> 0 but finite, so the non-finite
    midpoint fallback doesn't catch it). The result is a handful of long, thin
    triangles that bridge across the mesh and render as a spike. They are
    non-physical: in the otherwise uniform grid mesh their longest edge is
    ~100x the median. Removing them leaves small holes that the downstream
    ``_fill_holes`` / meshfix step closes. Device-agnostic (also benign on CUDA).
    """
    if faces.shape[0] == 0:
        return faces, 0
    tri = vertices[faces]
    emax = np.maximum.reduce([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ])
    med = float(np.median(emax))
    keep = emax <= edge_mult * med
    return faces[keep], int((~keep).sum())


def remove_small_components(vertices: np.array, faces: np.array, min_face_ratio: float = 0.01):
    """Keep only connected components with at least ``min_face_ratio`` of the largest
    component's face count. Removes the tiny single-triangle specks FlexiCubes leaves
    scattered around the body (and any residue from decimation), without assuming a
    single-component result."""
    if faces.shape[0] == 0:
        return vertices, faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return vertices, faces
    max_faces = max(c.faces.shape[0] for c in comps)
    keep = [c for c in comps if c.faces.shape[0] >= max(1, int(min_face_ratio * max_faces))]
    merged = trimesh.util.concatenate(keep)
    return np.asarray(merged.vertices), np.asarray(merged.faces)


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Remove FlexiCubes sliver/spike faces before anything else, so decimation and
    # hole-filling operate on a clean surface.
    faces, n_slivers = remove_sliver_faces(vertices, faces)
    if verbose and n_slivers:
        tqdm.write(f'Removed {n_slivers} sliver faces (spike)')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Remove invisible faces
    if fill_holes:
        vertices, faces = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        vertices, faces = _fill_holes(
            vertices, faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Drop leftover dust (tiny disconnected specks) and keep the meaningful body.
    vertices, faces = remove_small_components(vertices, faces)
    if verbose:
        tqdm.write(f'After small-component filter: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces


def barycentric_transfer_attributes(
    src_mesh: trimesh.Trimesh,
    src_attrs: np.ndarray,
    dst_vertices: np.ndarray,
) -> np.ndarray:
    """
    Transfer per-vertex attributes from a source mesh to new vertices via
    barycentric interpolation on the closest triangle.

    Args:
        src_mesh (trimesh.Trimesh): Source mesh (must have faces).
        src_attrs (np.ndarray): Per-vertex attributes on the source mesh. Shape (V_src, C).
        dst_vertices (np.ndarray): Destination vertex positions. Shape (V_dst, 3).

    Returns:
        np.ndarray: Interpolated attributes for each destination vertex. Shape (V_dst, C).
    """
    src_attrs = np.asarray(src_attrs, dtype=np.float64)
    dst_vertices = np.asarray(dst_vertices, dtype=np.float64)

    closest_points, _, triangle_ids = trimesh.proximity.closest_point(src_mesh, dst_vertices)

    face_indices = src_mesh.faces[triangle_ids]  # (N, 3)
    v0 = src_mesh.vertices[face_indices[:, 0]].astype(np.float64)
    v1 = src_mesh.vertices[face_indices[:, 1]].astype(np.float64)
    v2 = src_mesh.vertices[face_indices[:, 2]].astype(np.float64)

    # Barycentric coordinates via dot-product method
    e0 = v1 - v0
    e1 = v2 - v0
    w = closest_points.astype(np.float64) - v0

    d00 = np.sum(e0 * e0, axis=1)
    d01 = np.sum(e0 * e1, axis=1)
    d11 = np.sum(e1 * e1, axis=1)
    d20 = np.sum(w * e0, axis=1)
    d21 = np.sum(w * e1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    b1 = (d11 * d20 - d01 * d21) / denom
    b2 = (d00 * d21 - d01 * d20) / denom
    b0 = 1.0 - b1 - b2

    bary = np.stack([b0, b1, b2], axis=1)  # (N, 3)
    np.clip(bary, 0.0, None, out=bary)
    bary_sum = bary.sum(axis=1, keepdims=True)
    bary_sum = np.maximum(bary_sum, 1e-12)
    bary /= bary_sum

    a0 = src_attrs[face_indices[:, 0]]
    a1 = src_attrs[face_indices[:, 1]]
    a2 = src_attrs[face_indices[:, 2]]

    result = bary[:, 0:1] * a0 + bary[:, 1:2] * a1 + bary[:, 2:3] * a2
    return result.astype(np.float32)


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).

    Returns:
        vertices, faces, uvs, vmapping
        vmapping maps new vertex indices back to original vertex indices
        (new vertices may be duplicated at UV seams).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs, vmapping


@torch.no_grad()
def bake_vertex_colors_to_texture(
    dense_vertices: np.ndarray,
    dense_faces: np.ndarray,
    dense_vertex_colors: np.ndarray,
    simp_vertices: np.ndarray,
    simp_faces: np.ndarray,
    simp_uvs: np.ndarray,
    texture_size: int = 1024,
) -> np.ndarray:
    """
    Bake per-vertex colors from a dense mesh into a UV-mapped texture on a
    simplified mesh.

    For each texel covered by the simplified mesh in UV space, the 3D position
    is computed via nvdiffrast interpolation, then the closest point on the
    dense mesh is queried and its vertex color is barycentric-interpolated.

    Args:
        dense_vertices (np.ndarray): Dense mesh vertices. Shape (Vd, 3).
        dense_faces (np.ndarray): Dense mesh faces. Shape (Fd, 3).
        dense_vertex_colors (np.ndarray): Per-vertex RGB in [0,1]. Shape (Vd, 3).
        simp_vertices (np.ndarray): Simplified (UV-split) mesh vertices. Shape (Vs, 3).
        simp_faces (np.ndarray): Simplified mesh faces. Shape (Fs, 3).
        simp_uvs (np.ndarray): UV coordinates for simplified mesh. Shape (Vs, 2).
        texture_size (int): Output texture resolution (square).

    Returns:
        np.ndarray: Baked texture image, shape (texture_size, texture_size, 3), uint8.
    """
    device = _DEV
    verts_t = torch.tensor(simp_vertices, dtype=torch.float32, device=device)
    faces_t = torch.tensor(simp_faces.astype(np.int32), dtype=torch.int32, device=device)
    uvs_t = torch.tensor(simp_uvs, dtype=torch.float32, device=device)

    # Map UVs to clip space for nvdiffrast: [0,1] -> [-1,1], z=0, w=1
    uv_clip = torch.zeros(uvs_t.shape[0], 4, dtype=torch.float32, device=device)
    uv_clip[:, 0] = uvs_t[:, 0] * 2.0 - 1.0
    uv_clip[:, 1] = uvs_t[:, 1] * 2.0 - 1.0
    uv_clip[:, 2] = 0.0
    uv_clip[:, 3] = 1.0

    glctx = dr.RasterizeCudaContext()
    rast_out, _ = dr.rasterize(glctx, uv_clip[None], faces_t, resolution=[texture_size, texture_size])

    # Interpolate 3D positions at each texel
    pos_map, _ = dr.interpolate(verts_t[None].contiguous(), rast_out, faces_t)
    # pos_map: (1, H, W, 3)

    mask = (rast_out[0, :, :, 3] > 0)  # (H, W)
    positions = pos_map[0][mask].cpu().numpy()  # (N, 3)

    # Query dense mesh for closest-point colors
    dense_mesh = trimesh.Trimesh(vertices=dense_vertices, faces=dense_faces, process=False)
    closest_pts, _, tri_ids = trimesh.proximity.closest_point(dense_mesh, positions)

    face_verts = dense_faces[tri_ids]  # (N, 3)
    v0 = dense_vertices[face_verts[:, 0]]
    v1 = dense_vertices[face_verts[:, 1]]
    v2 = dense_vertices[face_verts[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0
    w = closest_pts - v0
    d00 = np.sum(e0 * e0, axis=1)
    d01 = np.sum(e0 * e1, axis=1)
    d11 = np.sum(e1 * e1, axis=1)
    d20 = np.sum(w * e0, axis=1)
    d21 = np.sum(w * e1, axis=1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    b1 = (d11 * d20 - d01 * d21) / denom
    b2 = (d00 * d21 - d01 * d20) / denom
    b0 = 1.0 - b1 - b2
    bary = np.stack([b0, b1, b2], axis=1)
    np.clip(bary, 0.0, None, out=bary)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)

    c0 = dense_vertex_colors[face_verts[:, 0]]
    c1 = dense_vertex_colors[face_verts[:, 1]]
    c2 = dense_vertex_colors[face_verts[:, 2]]
    colors = bary[:, 0:1] * c0 + bary[:, 1:2] * c1 + bary[:, 2:3] * c2

    # Write to texture (flip vertically to match image convention)
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    mask_np = mask.cpu().numpy()
    texture[mask_np] = colors.astype(np.float32)
    texture = np.flipud(texture)
    mask_np = np.flipud(mask_np)
    texture = np.clip(texture * 255, 0, 255).astype(np.uint8)

    inpaint_mask = (~mask_np).astype(np.uint8)
    texture = cv2.inpaint(texture, inpaint_mask, 3, cv2.INPAINT_TELEA)

    return texture


@torch.no_grad()
def render_multiview_mesh_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    resolution: int = 1024,
    nviews: int = 100,
    near: float = 0.1,
    far: float = 10.0,
    verbose: bool = True,
):
    """
    Render multiview color images from a mesh with per-vertex colors.

    Uses ``utils3d.torch.rasterize_triangle_faces`` — the exact same
    rasterisation path that :func:`bake_texture` uses internally — so
    the observations are guaranteed to be projection-aligned with the
    bake-texture rasterisation.

    Returns:
        observations: list of (H, W, 3) uint8 images in standard top-left origin
        extrinsics: list of numpy (4, 4)
        intrinsics: list of numpy (3, 3)
    """
    from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    r, fov = 2, 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [c[0] for c in cams], [c[1] for c in cams], r, fov,
    )

    verts_t = torch.tensor(vertices, dtype=torch.float32, device=_DEV)
    faces_t = torch.tensor(faces.astype(np.int32), dtype=torch.int32, device=_DEV)
    colors_t = torch.tensor(vertex_colors, dtype=torch.float32, device=_DEV).clamp(0, 1)

    rastctx = utils3d.torch.RastContext(backend=_rast_backend())
    observations = []

    for extr, intr in tqdm(
        zip(extrinsics, intrinsics), total=nviews,
        disable=not verbose, desc='Rendering multiview',
    ):
        view = utils3d.torch.extrinsics_to_view(extr)
        proj = utils3d.torch.intrinsics_to_perspective(intr, near, far)

        rast = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts_t[None], faces_t, resolution, resolution,
            uv=colors_t[None], view=view, projection=proj,
        )
        color_img = rast['uv'][0]   # (H, W, 3) interpolated vertex colors
        mask = rast['mask'][0] > 0.5

        # rasterisation is in OpenGL bottom-left origin; flip to top-left
        color_img = color_img.flip(0).clamp(0, 1)
        mask = mask.flip(0)
        # zero out background so mask-based workflows stay correct
        color_img[~mask] = 0

        observations.append(
            np.clip(color_img.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        )

    extrinsics_np = [e.cpu().numpy() for e in extrinsics]
    intrinsics_np = [i.cpu().numpy() for i in intrinsics]
    return observations, extrinsics_np, intrinsics_np


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """
    device = _DEV
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(np.int32), dtype=torch.int32, device=device)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    observations_cpu = [torch.tensor(obs / 255.0, dtype=torch.float32) for obs in observations]
    masks_cpu = [torch.tensor(m > 0, dtype=torch.bool) for m in masks]
    views_cpu = [utils3d.torch.extrinsics_to_view(torch.tensor(extr, dtype=torch.float32)).cpu() for extr in extrinsics]
    projections_cpu = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr, dtype=torch.float32), near, far).cpu() for intr in intrinsics]

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend=_rast_backend())
        for observation_cpu, mask_cpu, view_cpu, projection_cpu in tqdm(
            zip(observations_cpu, masks_cpu, views_cpu, projections_cpu),
            total=len(observations_cpu),
            disable=not verbose,
            desc='Texture baking (fast)',
        ):
            observation = observation_cpu.to(device)
            mask_src = mask_cpu.to(device)
            view = view_cpu.to(device)
            projection = projection_cpu.to(device)
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & mask_src
            
            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))
            del observation, mask_src, view, projection, rast, uv_map, mask, obs, idx

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # inpaint
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
        del texture_weights, rastctx
        _cuda_cleanup()

    elif mode == 'opt':
        rastctx = utils3d.torch.RastContext(backend=_rast_backend())
        observations_cpu = [obs.flip(0).contiguous() for obs in observations_cpu]
        masks_cpu = [m.flip(0).contiguous() for m in masks_cpu]
        _uv = []
        _uv_dr = []
        for observation_cpu, view_cpu, projection_cpu in tqdm(
            zip(observations_cpu, views_cpu, projections_cpu),
            total=len(views_cpu),
            disable=not verbose,
            desc='Texture baking (opt): UV',
        ):
            view = view_cpu.to(device)
            projection = projection_cpu.to(device)
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    vertices[None],
                    faces,
                    observation_cpu.shape[1],
                    observation_cpu.shape[0],
                    uv=uvs[None],
                    view=view,
                    projection=projection,
                )
                _uv.append(rast['uv'].detach().cpu())
                _uv_dr.append(rast['uv_dr'].detach().cpu())
            del view, projection, rast
        _cuda_cleanup()

        texture = torch.nn.Parameter(
            torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda()
        )
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))

        def tv_loss(texture):
            return torch.nn.functional.l1_loss(
                texture[:, :-1, :, :], texture[:, 1:, :, :]
            ) + torch.nn.functional.l1_loss(
                texture[:, :, :-1, :], texture[:, :, 1:, :]
            )

        total_steps = 500
        with tqdm(
            total=total_steps,
            disable=not verbose,
            desc='Texture baking (opt): optimizing',
        ) as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views_cpu))
                uv, uv_dr, observation, mask = (
                    _uv[selected].to(device),
                    _uv_dr[selected].to(device),
                    observations_cpu[selected].to(device),
                    masks_cpu[selected].to(device),
                )
                render = dr.texture(texture, uv, uv_dr)[0]
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                # annealing
                optimizer.param_groups[0]['lr'] = cosine_anealing(
                    optimizer, step, total_steps, 1e-2, 1e-5
                )
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
                del uv, uv_dr, observation, mask, render, loss
        del _uv, _uv_dr, optimizer
        _cuda_cleanup()

        texture = np.clip(
            texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
        del rastctx
        _cuda_cleanup()
    else:
        raise ValueError(f'Unknown mode: {mode}')

    del vertices, faces, uvs, observations_cpu, masks_cpu, views_cpu, projections_cpu
    _cuda_cleanup()
    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    # mesh postprocess
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )

    # parametrize mesh
    vertices, faces, uvs, _vmapping = parametrize_mesh(vertices, faces)

    # bake texture
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    texture = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    texture = Image.fromarray(texture)

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    return mesh


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.
    
    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs
    
    # simplify
    observations, extrinsics, intrinsics = render_multiview(gs, resolution=1024, nviews=100)
    observations = [torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1) for obs in observations]
    
    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer({
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0,0,0),
        })
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())
    
    start_lr = [1e-4, 1e-3, 5e-3, 0.025]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam([
        {"params": new_gs._xyz, "lr": start_lr[0]},
        {"params": new_gs._rotation, "lr": start_lr[1]},
        {"params": new_gs._scaling, "lr": start_lr[2]},
        {"params": new_gs._opacity, "lr": start_lr[3]},
    ], lr=start_lr[0])
    
    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
    
    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])
    
    with tqdm(total=2500, disable=not verbose, desc='Simplifying Gaussian') as pbar:
        for i in range(2500):
            # prune
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = new_gs._features_rest[mask] if new_gs._features_rest is not None else None
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # update optimizer state
                for param_group, new_param in zip(optimizer.param_groups, [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity]):
                    stored_state = optimizer.state[param_group['params'][0]]
                    if 'exp_avg' in stored_state:
                        stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                        stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                    del optimizer.state[param_group['params'][0]]
                    param_group['params'][0] = new_param
                    optimizer.state[param_group['params'][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()
            
            # sparisfy
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta
            
            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]
            
            color = renderer.render(new_gs, extrinsic, intrinsic)['color']
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            loss = rgb_loss + \
                   _delta * torch.sum(torch.pow(_lambda + opacity - _zeta, 2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]['lr'] = cosine_anealing(optimizer, i, 2500, start_lr[j], end_lr[j])
            
            pbar.set_postfix({'loss': rgb_loss.item(), 'num': opacity.shape[0], 'lambda': _lambda.mean().item()})
            pbar.update()
            
    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data
    
    return new_gs
