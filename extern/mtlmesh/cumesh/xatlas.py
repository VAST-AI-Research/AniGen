import torch
from typing import *
from tqdm import tqdm

try:
    from . import _xatlas
except ImportError:
    from . import _cumesh_xatlas as _xatlas

class Atlas:
    def __init__(self):
        self.atlas = _xatlas.Atlas()

    def add_mesh(self, 
                 vertices: torch.Tensor, 
                 faces: torch.Tensor, 
                 normals: Optional[torch.Tensor] = None, 
                 uvs: Optional[torch.Tensor] = None):
        """
        Add a mesh to the atlas.

        Args:
            vertices: [V, 3] float32 tensor of vertex positions.
            faces: [F, 3] int32 tensor of face indices.
            normals: Optional [V, 3] float32 tensor of vertex normals.
            uvs: Optional [V, 2] float32 tensor of input UVs (as hints).
        """
        # Validate vertices
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices must be [V, 3]"
        assert vertices.dtype == torch.float32, "vertices must be float32"
        assert vertices.device.type == 'cpu', "vertices must be on CPU"
        assert vertices.is_contiguous(), "vertices must be contiguous"

        # Validate faces
        assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be [F, 3]"
        assert faces.dtype == torch.int32, "faces must be int32"
        assert faces.device.type == 'cpu', "faces must be on CPU"
        assert faces.is_contiguous(), "faces must be contiguous"

        # Validate normals if present
        if normals is not None:
            assert normals.ndim == 2 and normals.shape[1] == 3, "normals must be [V, 3]"
            assert normals.dtype == torch.float32, "normals must be float32"
            assert normals.device.type == 'cpu', "normals must be on CPU"
            assert normals.is_contiguous(), "normals must be contiguous"
            assert normals.shape[0] == vertices.shape[0], "normals must have same length as vertices"

        # Validate UVs if present
        if uvs is not None:
            assert uvs.ndim == 2 and uvs.shape[1] == 2, "uvs must be [V, 2]"
            assert uvs.dtype == torch.float32, "uvs must be float32"
            assert uvs.device.type == 'cpu', "uvs must be on CPU"
            assert uvs.is_contiguous(), "uvs must be contiguous"
            assert uvs.shape[0] == vertices.shape[0], "uvs must have same length as vertices"

        self.atlas.add_mesh(vertices, faces, normals, uvs)

    def compute_charts(self, 
                       max_chart_area: float = 0.0,
                       max_boundary_length: float = 0.0,
                       normal_deviation_weight: float = 2.0,
                       roundness_weight: float = 0.01,
                       straightness_weight: float = 6.0,
                       normal_seam_weight: float = 4.0,
                       texture_seam_weight: float = 0.5,
                       max_cost: float = 2.0,
                       max_iterations: int = 1,
                       use_input_mesh_uvs: bool = False,
                       fix_winding: bool = False,
                       verbose: bool = False):
        """
        Compute charts (parameterization) for the added meshes.

        Args:
            max_chart_area: Don't grow charts to be larger than this. 0 means no limit.
            max_boundary_length: Don't grow charts to have a longer boundary than this. 0 means no limit.
            normal_deviation_weight: Weight for angle between face and average chart normal.
            roundness_weight: Weight for chart roundness.
            straightness_weight: Weight for chart straightness.
            normal_seam_weight: Weight for normal seams. If > 1000, normal seams are fully respected.
            texture_seam_weight: Weight for texture seams.
            max_cost: If total of all metrics * weights > maxCost, don't grow chart. Lower values result in more charts.
            max_iterations: Number of iterations of the chart growing and seeding phases. Higher values result in better charts.
            use_input_mesh_uvs: Use input UVs provided in add_mesh as hints.
            fix_winding: Enforce consistent texture coordinate winding.
            verbose: If True, display a progress bar (requires tqdm).
        """
        options = _xatlas.ChartOptions()
        options.max_chart_area = max_chart_area
        options.max_boundary_length = max_boundary_length
        options.normal_deviation_weight = normal_deviation_weight
        options.roundness_weight = roundness_weight
        options.straightness_weight = straightness_weight
        options.normal_seam_weight = normal_seam_weight
        options.texture_seam_weight = texture_seam_weight
        options.max_cost = max_cost
        options.max_iterations = max_iterations
        options.use_input_mesh_uvs = use_input_mesh_uvs
        options.fix_winding = fix_winding

        callback = self._create_progress_callback() if verbose else None
        self.atlas.compute_charts(options, callback)
        if verbose and self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def pack_charts(self, 
                    max_chart_size: int = 0,
                    padding: int = 0,
                    texels_per_unit: float = 0.0,
                    resolution: int = 0,
                    bilinear: bool = True,
                    block_align: bool = False,
                    brute_force: bool = False,
                    rotate_charts: bool = True,
                    rotate_charts_to_axis: bool = True,
                    verbose: bool = False):
        """
        Pack charts into the atlas texture.

        Args:
            max_chart_size: Charts larger than this will be scaled down. 0 means no limit.
            padding: Number of pixels to pad charts with.
            texels_per_unit: Unit to texel scale. If 0, estimated to match resolution.
            resolution: If not 0, generate atlas with this exact resolution (width=height).
            bilinear: Leave space around charts for bilinear filtering.
            block_align: Align charts to 4x4 blocks.
            brute_force: Slower, but gives best packing result.
            rotate_charts: Allow charts to be rotated to improve packing.
            rotate_charts_to_axis: Rotate charts to the axis of their convex hull.
            verbose: If True, display a progress bar (requires tqdm).
        """
        options = _xatlas.PackOptions()
        options.max_chart_size = max_chart_size
        options.padding = padding
        options.texels_per_unit = texels_per_unit
        options.resolution = resolution
        options.bilinear = bilinear
        options.block_align = block_align
        options.brute_force = brute_force
        options.rotate_charts = rotate_charts
        options.rotate_charts_to_axis = rotate_charts_to_axis

        callback = self._create_progress_callback() if verbose else None
        self.atlas.pack_charts(options, callback)
        if verbose and self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def get_mesh(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the result mesh data for a specific added mesh.
        
        Args:
            index: The index of the mesh (order of addition).

        Returns:
            A tuple containing:
            - xrefs: [NewV] int32 tensor mapping new vertices to original vertex indices.
            - faces: [NewF, 3] int32 tensor of the new face indices.
            - uvs: [NewV, 2] float32 tensor of the calculated UV coordinates.
        """
        return self.atlas.get_mesh(index)

    def _create_progress_callback(self):
        self._pbar = None
        self._current_category = ""

        def callback(category: str, progress: int) -> bool:
            # Check if category changed to reset/create bar
            if category != self._current_category:
                if self._pbar is not None:
                    self._pbar.close()
                self._current_category = category
                self._pbar = tqdm(total=100, desc=f"xatlas: {category}")
            
            # Update progress
            if self._pbar is not None:
                self._pbar.n = progress
                self._pbar.refresh()
            
            return True # Continue processing

        return callback
    