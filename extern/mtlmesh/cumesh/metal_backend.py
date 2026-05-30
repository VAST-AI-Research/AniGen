"""
MtlMesh: Metal-accelerated mesh processing for macOS.

Architecture:
- Per-element parallel work: Metal compute shaders (via _C extension)
- Collective ops (scan, sort, reduce): PyTorch MPS tensors
- All data lives on MPS device as torch tensors
"""
from typing import Tuple, Dict, Optional, List
import math
import torch
from tqdm import tqdm

# Import the native Metal extension
from . import _C


def _exclusive_sum(x: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum using PyTorch."""
    return torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), x.cumsum(0)[:-1]])


def _exclusive_sum_inplace(x: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum, returns result of same size."""
    cs = x.cumsum(0)
    result = torch.empty_like(x)
    result[0] = 0
    result[1:] = cs[:-1]
    return result


class MtlMesh:
    """
    Metal-accelerated mesh processor.
    All tensors are on MPS device.
    """
    def __init__(self):
        self._mesh = _C.MtlMesh()
        self._device = torch.device('mps')

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Initialize with vertices [V,3] float32 and faces [F,3] int32."""
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        v = vertices.contiguous().to(device=self._device, dtype=torch.float32)
        f = faces.contiguous().to(device=self._device, dtype=torch.int32)
        self._mesh.init(v, f)

    @property
    def num_vertices(self) -> int:
        return self._mesh.num_vertices()

    @property
    def num_faces(self) -> int:
        return self._mesh.num_faces()

    @property
    def num_edges(self) -> int:
        return self._mesh.num_edges()

    @property
    def num_boundaries(self) -> int:
        return self._mesh.num_boundaries()

    @property
    def num_conneted_components(self) -> int:
        return self._mesh.num_conneted_components()

    @property
    def num_boundary_conneted_components(self) -> int:
        return self._mesh.num_boundary_conneted_components()

    @property
    def num_boundary_loops(self) -> int:
        return self._mesh.num_boundary_loops()

    def clear_cache(self):
        self._mesh.clear_cache()

    def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mesh.read()

    def read_face_areas(self) -> torch.Tensor:
        return self._mesh.read_face_areas()

    def read_face_normals(self) -> torch.Tensor:
        return self._mesh.read_face_normals()

    def read_vertex_normals(self) -> torch.Tensor:
        return self._mesh.read_vertex_normals()

    def read_edges(self) -> torch.Tensor:
        return self._mesh.read_edges()

    def read_boundaries(self) -> torch.Tensor:
        return self._mesh.read_boundaries()

    def read_manifold_face_adjacency(self) -> torch.Tensor:
        return self._mesh.read_manifold_face_adjacency()

    def read_manifold_boundary_adjacency(self) -> torch.Tensor:
        return self._mesh.read_manifold_boundary_adjacency()

    def read_connected_components(self) -> Tuple[int, torch.Tensor]:
        return self._mesh.read_connected_components()

    def read_boundary_connected_components(self) -> Tuple[int, torch.Tensor]:
        return self._mesh.read_boundary_connected_components()

    def read_boundary_loops(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return self._mesh.read_boundary_loops()

    def read_all_cache(self) -> Dict[str, torch.Tensor]:
        return self._mesh.read_all_cache()

    def compute_face_areas(self):
        self._mesh.compute_face_areas()

    def compute_face_normals(self):
        self._mesh.compute_face_normals()

    def compute_vertex_normals(self):
        self._mesh.compute_vertex_normals()

    def get_vertex_face_adjacency(self):
        self._mesh.get_vertex_face_adjacency()

    def get_edges(self):
        self._mesh.get_edges()

    def get_edge_face_adjacency(self):
        self._mesh.get_edge_face_adjacency()

    def get_vertex_edge_adjacency(self):
        self._mesh.get_vertex_edge_adjacency()

    def get_boundary_info(self):
        self._mesh.get_boundary_info()

    def get_vertex_boundary_adjacency(self):
        self._mesh.get_vertex_boundary_adjacency()

    def get_vertex_is_manifold(self):
        self._mesh.get_vertex_is_manifold()

    def get_manifold_face_adjacency(self):
        self._mesh.get_manifold_face_adjacency()

    def get_manifold_boundary_adjacency(self):
        self._mesh.get_manifold_boundary_adjacency()

    def get_connected_components(self):
        self._mesh.get_connected_components()

    def get_boundary_connected_components(self):
        self._mesh.get_boundary_connected_components()

    def get_boundary_loops(self):
        self._mesh.get_boundary_loops()

    def remove_faces(self, face_mask: torch.Tensor):
        assert face_mask.dtype == torch.bool
        self._mesh.remove_faces(face_mask.to(self._device))

    def remove_unreferenced_vertices(self):
        self._mesh.remove_unreferenced_vertices()

    def remove_duplicate_faces(self):
        self._mesh.remove_duplicate_faces()

    def remove_degenerate_faces(self, abs_thresh: float = 1e-24, rel_thresh: float = 1e-12):
        self._mesh.remove_degenerate_faces(abs_thresh, rel_thresh)

    def fill_holes(self, max_hole_perimeter: float = 3e-2):
        self._mesh.fill_holes(max_hole_perimeter)

    def repair_non_manifold_edges(self):
        self._mesh.repair_non_manifold_edges()

    def remove_non_manifold_faces(self):
        self._mesh.remove_non_manifold_faces()

    def remove_small_connected_components(self, min_area: float):
        self._mesh.remove_small_connected_components(min_area)

    def unify_face_orientations(self):
        self._mesh.unify_face_orientations()

    def simplify_step(self, lambda_edge_length: float, lambda_skinny: float, threshold: float, timing: bool = False):
        return self._mesh.simplify_step(lambda_edge_length, lambda_skinny, threshold, timing)

    def simplify(self, target_num_faces: int, verbose: bool = False, options: dict = {}):
        """Simplify mesh using QEM-based parallel edge collapse."""
        assert isinstance(target_num_faces, int) and target_num_faces > 0

        num_face = self.num_faces
        if num_face <= target_num_faces:
            return

        if verbose:
            pbar = tqdm(total=num_face - target_num_faces, desc="Simplifying")

        thresh = options.get('thresh', 1e-8)
        lambda_edge_length = options.get('lambda_edge_length', 1e-2)
        lambda_skinny = options.get('lambda_skinny', 1e-3)

        while True:
            if verbose:
                pbar.set_description(f"Simplifying [thres={thresh:.2e}]")

            new_num_vert, new_num_face = self.simplify_step(
                lambda_edge_length, lambda_skinny, thresh, False
            )

            if verbose:
                pbar.update(num_face - max(target_num_faces, new_num_face))

            if new_num_face <= target_num_faces:
                break

            del_num_face = num_face - new_num_face
            if del_num_face / num_face < 1e-2:
                thresh *= 10
            num_face = new_num_face

        if verbose:
            pbar.close()

    def compute_charts(
        self,
        threshold_cone_half_angle_rad: float = math.radians(90),
        refine_iterations: int = 100,
        global_iterations: int = 3,
        smooth_strength: float = 1,
        area_penalty_weight: float = 0.1,
        perimeter_area_ratio_weight: float = 0.0001,
    ):
        self._mesh.compute_charts(
            threshold_cone_half_angle_rad,
            refine_iterations,
            global_iterations,
            smooth_strength,
            area_penalty_weight,
            perimeter_area_ratio_weight
        )

    def read_atlas_charts(self):
        return self._mesh.read_atlas_charts()

    def uv_unwrap(
        self,
        compute_charts_kwargs: dict = {},
        xatlas_compute_charts_kwargs: dict = {},
        xatlas_pack_charts_kwargs: dict = {},
        return_vmaps: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """UV unwrap using fast chart clustering + xatlas packing."""
        from .xatlas import Atlas

        xatlas_compute_charts_kwargs['verbose'] = verbose
        xatlas_pack_charts_kwargs['verbose'] = verbose

        self.remove_degenerate_faces()

        # 1. Fast mesh clustering
        self.compute_charts(**compute_charts_kwargs)
        new_vertices, new_faces = self.read()
        num_charts, charts_id, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = self.read_atlas_charts()
        chart_vertices = new_vertices[chart_vmap].cpu()
        chart_faces = chart_faces.cpu()
        chart_vertex_offset = chart_vertex_offset.cpu()
        chart_face_offset = chart_face_offset.cpu()
        chart_vmap = chart_vmap.cpu()
        if verbose:
            print(f"Get {num_charts} clusters after fast clustering")

        # 2. Xatlas packing
        xatlas = Atlas()
        chart_vmaps = []
        for i in tqdm(range(num_charts), desc="Adding clusters to xatlas", disable=not verbose):
            chart_faces_i = chart_faces[chart_face_offset[i]:chart_face_offset[i+1]] - chart_vertex_offset[i]
            chart_vertices_i = chart_vertices[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            chart_vmap_i = chart_vmap[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            chart_vmaps.append(chart_vmap_i)
            xatlas.add_mesh(chart_vertices_i, chart_faces_i)
        xatlas.compute_charts(**xatlas_compute_charts_kwargs)
        xatlas.pack_charts(**xatlas_pack_charts_kwargs)

        vmaps = []
        faces = []
        uvs = []
        cnt = 0
        for i in tqdm(range(num_charts), desc="Gathering results from xatlas", disable=not verbose):
            vmap, x_faces, x_uvs = xatlas.get_mesh(i)
            vmaps.append(chart_vmaps[i][vmap])
            faces.append(x_faces + cnt)
            uvs.append(x_uvs)
            cnt += vmap.shape[0]
        vmaps = torch.cat(vmaps, dim=0)
        vertices = new_vertices.cpu()[vmaps]
        faces = torch.cat(faces, dim=0)
        uvs = torch.cat(uvs, dim=0)

        out = [vertices, faces, uvs]
        if return_vmaps:
            out.append(vmaps)
        return tuple(out)
