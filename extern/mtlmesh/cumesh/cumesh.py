from typing import *
import math
import torch
from tqdm import tqdm
from .xatlas import Atlas
from . import _C


class CuMesh:
    def __init__(self):
        self.cu_mesh = _C.CuMesh()

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Initialize the CuMesh with vertices and faces.

        Args:
            vertices: a tensor of shape [V, 3] containing the vertex positions.
            faces: a tensor of shape [F, 3] containing the face indices.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Input vertices must be of shape [V, 3]"
        assert faces.ndim == 2 and faces.shape[1] == 3, "Input faces must be of shape [F, 3]"
        assert vertices.is_contiguous() and faces.is_contiguous(), "Input tensors must be contiguous"
        assert vertices.is_cuda and faces.is_cuda and vertices.device == faces.device, "Input tensors must both be on the same CUDA device"
        self.cu_mesh.init(vertices, faces)
        
    @property
    def num_vertices(self) -> int:
        return self.cu_mesh.num_vertices()
    
    @property
    def num_faces(self) -> int:
        return self.cu_mesh.num_faces()
    
    @property
    def num_edges(self) -> int:
        return self.cu_mesh.num_edges()
    
    @property
    def num_boundaries(self) -> int:
        return self.cu_mesh.num_boundaries()
    
    @property
    def num_conneted_components(self) -> int:
        return self.cu_mesh.num_conneted_components()
    
    @property
    def num_boundary_conneted_components(self) -> int:
        return self.cu_mesh.num_boundary_conneted_components()
    
    @property
    def num_boundary_loops(self) -> int:
        return self.cu_mesh.num_boundary_loops()

    def clear_cache(self):
        """
        Clear the cached data.
        """
        self.cu_mesh.clear_cache()

    def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read the current vertices and faces from the CuMesh.

        Returns:
            A tuple of two tensors: the vertex positions and the face indices.
        """
        return self.cu_mesh.read()
    
    def read_face_normals(self) -> torch.Tensor:
        """
        Read the normals of the faces from the CuMesh.

        Returns:
            The face normals as an [F, 3] tensor.
        """
        return self.cu_mesh.read_face_normals()
    
    def read_vertex_normals(self) -> torch.Tensor:
        """
        Read the normals of the vertices from the CuMesh.

        Returns:
            The vertex normals as an [V, 3] tensor.
        """
        return self.cu_mesh.read_vertex_normals()
    
    def read_edges(self) -> torch.Tensor:
        """
        Read the edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [E, 2] containing the edge indices.
        """
        return self.cu_mesh.read_edges()
    
    def read_boundaries(self) -> torch.Tensor:
        """
        Read the boundary edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [B] containing the boundary edge indices.
        """
        return self.cu_mesh.read_boundaries()
    
    
    def read_manifold_face_adjacency(self) -> torch.Tensor:
        """
        Read the manifold face adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold face adjacency.
        """
        return self.cu_mesh.read_manifold_face_adjacency()
    
    def read_manifold_boundary_adjacency(self) -> torch.Tensor:
        """
        Read the manifold boundary adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold boundary adjacency.
        """
        return self.cu_mesh.read_manifold_boundary_adjacency()
    
    def read_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each face.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [F] containing the connected component ID for each face.
        """
        return self.cu_mesh.read_connected_components()
    
    def read_boundary_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each boundary edge.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [E] containing the connected component ID for each boundary edge.
        """
        return self.cu_mesh.read_boundary_connected_components()
    
    def read_boundary_loops(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Read the boundary loops of the mesh.

        Returns:
            A tuple of three values:
                - the number of boundary loops
                - a tensor of shape [L] containing the indices of the boundary edges in each loop.
                - a tensor of shape [N_loops + 1] containing the offsets of the boundary edges in each loop.
        """
        return self.cu_mesh.read_boundary_loops()
    
    def read_all_cache(self) -> Dict[str, torch.Tensor]:
        """
        Read all cached data.

        Returns:
            A dictionary of cached data.
        """
        return self.cu_mesh.read_all_cache()
    
    def compute_face_normals(self):
        """
        Compute the normals of the faces.
        """
        self.cu_mesh.compute_face_normals()
    
    def compute_vertex_normals(self):
        """
        Compute the normals of the vertices.
        """
        self.cu_mesh.compute_vertex_normals()
        
    def get_vertex_face_adjacency(self):
        """
        Compute the vertex to face adjacency.
        """
        self.cu_mesh.get_vertex_face_adjacency()
        
    def get_edges(self):
        """
        Compute the edges of the mesh.
        """
        self.cu_mesh.get_edges()
        
    def get_edge_face_adjacency(self):
        """
        Compute the edge to face adjacency.
        """
        self.cu_mesh.get_edge_face_adjacency()
        
    def get_vertex_edge_adjacency(self):
        """
        Compute the vertex to edge adjacency.
        """
        self.cu_mesh.get_vertex_edge_adjacency()
        
    def get_boundary_info(self):
        """
        Compute the boundary information of the mesh.
        """
        self.cu_mesh.get_boundary_info()
        
    def get_vertex_boundary_adjacency(self):
        """
        Compute the vertex to boundary adjacency.
        """
        self.cu_mesh.get_vertex_boundary_adjacency()
        
    def get_manifold_face_adjacency(self):
        """
        Compute the manifold face adjacency.
        """
        self.cu_mesh.get_manifold_face_adjacency()
        
    def get_manifold_boundary_adjacency(self):
        """
        Compute the manifold boundary adjacency.
        """
        self.cu_mesh.get_manifold_boundary_adjacency()
        
    def get_connected_components(self):
        """
        Compute the connected components of the mesh.
        """
        self.cu_mesh.get_connected_components()
        
    def get_boundary_connected_components(self):
        """
        Compute the connected components of the boundary of the mesh.
        """
        self.cu_mesh.get_boundary_connected_components()
        
    def get_boundary_loops(self):
        """
        Compute the boundary loops of the mesh.
        """
        self.cu_mesh.get_boundary_loops()
        
    def remove_faces(self, face_mask: torch.Tensor):
        """
        Remove faces from the mesh.

        Args:
            face_mask: a boolean tensor of shape [F] indicating which faces to remove.
        """
        assert face_mask.ndim == 1 and face_mask.shape[0] == self.num_faces, "face_mask must be a boolean tensor of shape [F]"
        assert face_mask.is_contiguous() and face_mask.is_cuda, "face_mask must be a CUDA tensor"
        assert face_mask.dtype == torch.bool, "face_mask must be a boolean tensor"
        self.cu_mesh.remove_faces(face_mask)
    
    def remove_unreferenced_vertices(self):
        """
        Remove unreferenced vertices from the mesh.
        """
        self.cu_mesh.remove_unreferenced_vertices()
        
    def remove_duplicate_faces(self):
        """
        Remove duplicate faces from the mesh.
        """
        self.cu_mesh.remove_duplicate_faces()
        
    def remove_degenerate_faces(self, abs_thresh: float=1e-24, rel_thresh: float=1e-12):
        """
        Remove degenerate faces from the mesh.

        Args:
            abs_thresh: absolute area threshold below which a face is considered degenerate.
            rel_thresh: relative area to square of the longest edge threshold below which a face is considered degenerate.
                Note that a face is considered degenerate if both the absolute and relative conditions are met.
        """
        self.cu_mesh.remove_degenerate_faces(abs_thresh, rel_thresh)
        
    def fill_holes(self, max_hole_perimeter: float=3e-2):
        """
        Fill holes in the mesh.

        Args:
            max_hole_perimeter: the maximum perimeter of a hole to fill.
        """
        self.cu_mesh.fill_holes(max_hole_perimeter)
        
    def repair_non_manifold_edges(self):
        """
        Repair Non-manifold edges by splitting vertices.
        This creates duplicate vertices with the same coordinates.
        """
        self.cu_mesh.repair_non_manifold_edges()

    def remove_non_manifold_faces(self):
        """
        Remove faces on non-manifold edges.
        For each non-manifold edge (shared by >2 faces), only keep the first 2 faces.
        This repairs non-manifold edges by deleting faces instead of splitting vertices.
        """
        self.cu_mesh.remove_non_manifold_faces()
        
    def remove_small_connected_components(self, min_area: float):
        """
        Repair Non-manifold edges by splitting edges
        
        Args:
            min_area: the minimum area of a connected component to keep.
        """
        self.cu_mesh.remove_small_connected_components(min_area)
        
    def unify_face_orientations(self):
        """
        Unify the orientations of the faces.
        """
        self.cu_mesh.unify_face_orientations()
    
    def simplify(self, target_num_faces: int, verbose: bool=False, options: dict={}):
        """
        Simplifies the mesh using a fast approximation algorithm with gpu acceleration.

        Args:
            target_num_faces: the target number of faces to simplify to.
            verbose: whether to print the progress of the simplification.
            options: a dictionary of options for the simplification algorithm.
        """
        assert isinstance(target_num_faces, int) and target_num_faces > 0, "target_num_faces must be a positive integer"

        num_face = self.cu_mesh.num_faces()
        if num_face <= target_num_faces:
            return
        
        if verbose:
            pbar = tqdm(total=num_face-target_num_faces, desc="Simplifying", disable=not verbose)

        thresh = options.get('thresh', 1e-8)
        lambda_edge_length = options.get('lambda_edge_length', 1e-2)
        lambda_skinny = options.get('lambda_skinny', 1e-3)
        while True:
            if verbose:
                pbar.set_description(f"Simplifying [thres={thresh:.2e}]")
            
            new_num_vert, new_num_face = self.cu_mesh.simplify_step(lambda_edge_length, lambda_skinny, thresh, False)
            
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
        threshold_cone_half_angle_rad: float=math.radians(90),
        refine_iterations: int=100,
        global_iterations: int=3,
        smooth_strength: float=1,
        area_penalty_weight: float=0.1,
        perimeter_area_ratio_weight: float=0.0001,
    ):
        """
        Compute the atlas charts.

        Args:
            threshold_cone_half_angle_rad: The threshold for the cone half angle in radians.
            refine_iterations: The number of refinement iterations.
            smooth_strength: The strength of chart boundary smoothing.
            area_penalty_weight: Coefficient for chart size penalty. Cost += Area * weight.
                                 Prevents charts from becoming too large if > 0, 
                                 or encourages larger charts if < 0 (though usually used to penalize size variance).
            perimeter_area_ratio_weight: Coefficient for shape irregularity (long-strip) penalty. 
                                         Cost += (Perimeter / Area) * weight.
                                         Higher values penalize long strips and encourage circular/compact shapes.
        """
        self.cu_mesh.compute_charts(
            threshold_cone_half_angle_rad,
            refine_iterations,
            global_iterations,
            smooth_strength,
            area_penalty_weight,
            perimeter_area_ratio_weight
        )
        
    def read_atlas_charts(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read the atlas chart IDs for each face.
        
        Returns:
            A tuple of two values:
                - the number of charts
                - a tensor of shape [F] containing the chart ID for each face.
                - a tensor of shape [V] containing the vertex map
                - a tensor of shape [F, 3] containing the chart faces
                - a tensor of shape [C+1] containing the offsets of the chart vertices in the vertices tensor.
                - a tensor of shape [C+1] containing the offsets of the chart faces in the faces tensor.
        """
        return self.cu_mesh.read_atlas_charts()
    
    def uv_unwrap(
        self,
        compute_charts_kwargs: dict={},
        xatlas_compute_charts_kwargs: dict={},
        xatlas_pack_charts_kwargs: dict={},
        return_vmaps: bool=False,
        verbose: bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameterize the mesh using the accelerated mesh clustering and Xatlas
        
        Args:
            compute_charts_kwargs: a dictionary of options for the compute_charts function.
            xatlas_compute_charts_kwargs: a dictionary of options for the xatlas compute_charts function.
            xatlas_pack_charts_kwargs: a dictionary of options for the xatlas pack_charts function.
            return_vmaps: whether to return the vertex maps.
            verbose: whether to print the progress.
            
        Returns:
            A tuple of:
                - the vertex positions
                - the face indices
                - the uv coordinates
                - (optional) the map from the new vertex indices to the old vertex indices
        """
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
            