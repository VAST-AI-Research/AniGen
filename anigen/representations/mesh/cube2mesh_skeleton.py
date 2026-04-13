import math
import torch
from typing import Optional
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes
from pytorch3d.ops import knn_points


class AniGenMeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        vertex_skin_feats=None,
        grid_positions=None,
        grid_skin_feats=None,
        res=64,
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.vertex_skin_feats = vertex_skin_feats
        self.grid_positions = grid_positions
        self.grid_skin_feats = grid_skin_feats
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   


class AniGenSparseFeatures2Mesh:
    def __init__(
        self,
        device="cuda",
        res=64,
        use_color=True,
        skin_feat_channels=32,
        predict_skin=True,
        interpolate_skin_sparse=False,
        use_nearest_skin_feat=False,
        vertex_skin_feat_interp_sparse: Optional[bool] = None,
        vertex_skin_feat_interp_nearest: Optional[bool] = None,
        vertex_skin_feat_interp_use_deformed_grid: bool = False,
        vertex_skin_feat_interp_trilinear: bool = False,
        flexicube_disable_deform: bool = False,
        vertex_skin_feat_nodeform_trilinear: bool = False,
    ):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device, use_color=use_color)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self.skin_feat_channels = skin_feat_channels if predict_skin else 0
        self.predict_skin = predict_skin

        # Backward-compatible aliasing.
        if vertex_skin_feat_interp_sparse is None:
            vertex_skin_feat_interp_sparse = interpolate_skin_sparse
        if vertex_skin_feat_interp_nearest is None:
            vertex_skin_feat_interp_nearest = use_nearest_skin_feat

        self.vertex_skin_feat_interp_sparse = bool(vertex_skin_feat_interp_sparse)
        self.vertex_skin_feat_interp_nearest = bool(vertex_skin_feat_interp_nearest)
        self.vertex_skin_feat_interp_use_deformed_grid = bool(vertex_skin_feat_interp_use_deformed_grid)
        self.vertex_skin_feat_interp_trilinear = bool(vertex_skin_feat_interp_trilinear)
        self.flexicube_disable_deform = bool(flexicube_disable_deform)
        self.vertex_skin_feat_nodeform_trilinear = bool(vertex_skin_feat_nodeform_trilinear)

        # Combined mode overrides individual toggles.
        if self.vertex_skin_feat_nodeform_trilinear:
            # Force deform off and use strict trilinear interpolation.
            self.flexicube_disable_deform = True
            self.vertex_skin_feat_interp_trilinear = True
            self.vertex_skin_feat_interp_use_deformed_grid = False
            self.vertex_skin_feat_interp_nearest = False
            # Ensure we take the sparse-interp branch (i.e., don't read skin from FlexiCubes colors).
            self.vertex_skin_feat_interp_sparse = True

        self.interpolate_skin_sparse = self.vertex_skin_feat_interp_sparse and predict_skin and self.skin_feat_channels > 0
        self.use_nearest_skin_feat = self.vertex_skin_feat_interp_nearest
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = [
            ('sdf', {'shape': (8, 1), 'size': 8}),
            ('deform', {'shape': (8, 3), 'size': 8 * 3}),
            ('weights', {'shape': (21,), 'size': 21}),
        ]
        if self.use_color:
            # 6 channel color including normal map
            LAYOUTS.append(('color', {'shape': (8, 6,), 'size': 8 * 6}))
        if self.predict_skin:
            # Ensure skin_feat is always at the end
            LAYOUTS.append(('skin_feat', {'shape': (8, self.skin_feat_channels,), 'size': 8*self.skin_feat_channels}))
        self.layouts = edict()
        start = 0
        for k, v in LAYOUTS:
            v['range'] = (start, start + v['size'])
            self.layouts[k] = v
            start += v['size']
        # Do not include skin_feat in feats_channels if not predicting skin
        self.feats_channels = start - 8*self.skin_feat_channels if self.predict_skin else start
        self.skin_feat_channels = self.skin_feat_channels * 8 if self.predict_skin else 0
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])

    def _interpolate_skin_features(self, vertices, grid_points, grid_features, res):
        if grid_features is None or grid_features.shape[1] == 0:
            return None
        device = vertices.device
        feat_dtype = grid_features.dtype

        grid_points = grid_points.to(device=device, dtype=torch.float32)
        # Backward compatibility: if integer grid coords are passed, normalize to [-0.5, 0.5].
        if grid_points.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) or grid_points.abs().max() > 1.5:
            grid_points = (grid_points + 0.5) / res - 0.5
        # IMPORTANT for training stability: do not backprop through coordinates/distances.
        grid_points = grid_points.detach()
        vertex_points = vertices.to(device=device, dtype=torch.float32).detach()

        k = 1 if self.use_nearest_skin_feat else min(8, grid_points.shape[0])
        if k == 0:
            return torch.zeros(vertices.shape[0], grid_features.shape[1], device=device, dtype=feat_dtype)

        dist2, idx, _ = knn_points(vertex_points.unsqueeze(0), grid_points.unsqueeze(0), K=k, return_nn=False)
        
        if self.use_nearest_skin_feat:
            idx = idx[0, :, 0]
            return grid_features[idx].to(device=device, dtype=feat_dtype)

        dist = torch.sqrt(dist2[0]).clamp_min(1e-12)
        idx = idx[0]

        feats = grid_features.to(device=device, dtype=feat_dtype)
        neighbor_feats = feats[idx]
        # Smooth kernel weights to reduce jitter from neighbor swaps.
        # Grid spacing in normalized coords is ~ 1/res.
        sigma = (1.5 / res)
        weights = torch.exp(-(dist ** 2) / (2.0 * (sigma ** 2)))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights.to(dtype=neighbor_feats.dtype)

        vertex_skin_feats = torch.sum(neighbor_feats * weights.unsqueeze(-1), dim=1)
        return vertex_skin_feats

    def _interpolate_skin_features_trilinear(self, vertices, grid_features_dense, res: int):
        """Trilinear interpolation from regular (res+1)^3 grid vertices.

        This is deform-independent as long as `vertices` are in the same canonical
        coordinate system as `get_defomed_verts(..., deform=0)` i.e. v/res - 0.5.
        """
        if grid_features_dense is None or grid_features_dense.shape[-1] == 0:
            return None
        device = vertices.device
        feat_dtype = grid_features_dense.dtype

        # grid_features_dense: [(res+1)^3, C] -> [res+1, res+1, res+1, C]
        C = grid_features_dense.shape[-1]
        grid = grid_features_dense.view(res + 1, res + 1, res + 1, C).to(device=device)

        # vertices are in [-0.5, 0.5]; map to grid coordinate in [0, res]
        v = vertices.to(device=device, dtype=torch.float32).detach()
        g = (v + 0.5) * float(res)
        # Clamp so that i0+1 is always valid.
        eps = 1e-6
        g = torch.clamp(g, 0.0, float(res) - eps)

        i0 = torch.floor(g[:, 0]).to(torch.long)
        j0 = torch.floor(g[:, 1]).to(torch.long)
        k0 = torch.floor(g[:, 2]).to(torch.long)
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        tx = (g[:, 0] - i0.to(g.dtype)).unsqueeze(-1)
        ty = (g[:, 1] - j0.to(g.dtype)).unsqueeze(-1)
        tz = (g[:, 2] - k0.to(g.dtype)).unsqueeze(-1)

        def gather(ii, jj, kk):
            return grid[ii, jj, kk]

        c000 = gather(i0, j0, k0)
        c100 = gather(i1, j0, k0)
        c010 = gather(i0, j1, k0)
        c110 = gather(i1, j1, k0)
        c001 = gather(i0, j0, k1)
        c101 = gather(i1, j0, k1)
        c011 = gather(i0, j1, k1)
        c111 = gather(i1, j1, k1)

        wx0 = 1.0 - tx
        wy0 = 1.0 - ty
        wz0 = 1.0 - tz

        out = (
            c000 * (wx0 * wy0 * wz0) +
            c100 * (tx  * wy0 * wz0) +
            c010 * (wx0 * ty  * wz0) +
            c110 * (tx  * ty  * wz0) +
            c001 * (wx0 * wy0 * tz ) +
            c101 * (tx  * wy0 * tz ) +
            c011 * (wx0 * ty  * tz ) +
            c111 * (tx  * ty  * tz )
        )
        return out.to(dtype=feat_dtype)
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        
        skin_feat_channels = self.skin_feat_channels // 8 if self.predict_skin else 0

        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform]
        if self.predict_skin:
            skin_feat = self.get_layout(feats, 'skin_feat')
            v_attrs.append(skin_feat)
        if self.use_color:
            v_attrs.append(torch.sigmoid(color))
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        # Grid-vertex canonical coordinates in [-0.5, 0.5] (consistent with deform=0).
        v_pos_normalized = v_pos / self.res - 0.5
        grid_skin_feats = v_attrs[:, 4:4+skin_feat_channels] if self.predict_skin and skin_feat_channels > 0 else None
        if self.predict_skin and self.interpolate_skin_sparse and skin_feat_channels > 0:
            v_attrs_for_dense = torch.cat([v_attrs[:, :4], v_attrs[:, 4+skin_feat_channels:]], dim=-1)
        else:
            v_attrs_for_dense = v_attrs
        v_attrs_d = get_dense_attrs(v_pos, v_attrs_for_dense, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)

        sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        deform_d_eff = deform_d if not self.flexicube_disable_deform else (deform_d * 0.0)
        x_nx3 = get_defomed_verts(self.reg_v, deform_d_eff, self.res).type(sdf_d.dtype)
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training,
            no_sigmoid=True)

        rgbnormal_colors = None
        vertex_skin_feats = None
        start = 0
        if self.predict_skin and skin_feat_channels > 0:
            if self.interpolate_skin_sparse:
                # Deform-independent trilinear interpolation from 8 grid-vertex features.
                if self.vertex_skin_feat_nodeform_trilinear:
                    grid_features_dense = get_dense_attrs(v_pos, grid_skin_feats, res=self.res+1, sdf_init=False)
                    vertex_skin_feats = self._interpolate_skin_features_trilinear(
                        vertices=vertices,
                        grid_features_dense=grid_features_dense,
                        res=self.res,
                    )
                # Backward-compatible: allow trilinear if explicitly enabled alongside deform-disable.
                elif self.vertex_skin_feat_interp_trilinear and self.flexicube_disable_deform:
                    grid_features_dense = get_dense_attrs(v_pos, grid_skin_feats, res=self.res+1, sdf_init=False)
                    vertex_skin_feats = self._interpolate_skin_features_trilinear(
                        vertices=vertices,
                        grid_features_dense=grid_features_dense,
                        res=self.res,
                    )
                else:
                    # Choose the coordinate space used for KNN distances.
                    # - Regular grid space: stable, independent of predicted deformation.
                    # - Deformed space: matches mesh vertices but depends on deformation prediction.
                    if self.vertex_skin_feat_interp_use_deformed_grid:
                        grid_points_for_skin = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
                    else:
                        grid_points_for_skin = v_pos / self.res - 0.5
                    grid_features_for_skin = grid_skin_feats

                    vertex_skin_feats = self._interpolate_skin_features(
                        vertices=vertices,
                        grid_points=grid_points_for_skin,
                        grid_features=grid_features_for_skin,
                        res=self.res,
                    )
            else:
                vertex_skin_feats = colors[:, start: start + skin_feat_channels]
                start += skin_feat_channels
        if self.use_color:
            if colors is not None and colors.shape[1] >= start + 6:
                rgbnormal_colors = colors[:, start: start + 6]
            elif colors is not None and colors.shape[1] >= 6:
                rgbnormal_colors = colors[:, -6:]
            else:
                rgbnormal_colors = None
        
        mesh = AniGenMeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=rgbnormal_colors, vertex_skin_feats=vertex_skin_feats, grid_positions=v_pos_normalized, grid_skin_feats=grid_skin_feats, res=self.res)
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        
        return mesh


class AniGenSklFeatures2Skeleton:
    def __init__(self, skin_feat_channels=32, device="cuda", res=64, use_conf_jp=False, use_conf_skin=False, predict_skin=True, defined_on_center=False, jp_hyper_continuous=False, jp_residual_fields=False):
        self.device=device
        self.res = res
        self.use_conf_jp = use_conf_jp or jp_hyper_continuous
        self.jp_hyper_continuous = jp_hyper_continuous
        self.jp_residual_fields = jp_residual_fields
        self.use_conf_skin = use_conf_skin and not jp_hyper_continuous
        self.predict_skin = predict_skin
        self.skin_feat_channels = skin_feat_channels if predict_skin else 0
        self.defined_on_center = defined_on_center
        self._calc_layout()
    
    def _calc_layout(self):
        if self.defined_on_center:
            LAYOUTS = {
                'joint': {'shape': (3,), 'size': 3},
                'parent': {'shape': (3,), 'size': 3},
            }
            if self.use_conf_jp:
                LAYOUTS['conf_j'] = {'shape': (1,), 'size': 1}
                LAYOUTS['conf_p'] = {'shape': (1,), 'size': 1}
            if self.use_conf_skin:
                LAYOUTS['conf_skin'] = {'shape': (1,), 'size': 1}
            # Define skin features at the end
            if self.predict_skin:
                LAYOUTS['skin_feat'] = {'shape': (self.skin_feat_channels,), 'size': self.skin_feat_channels}
        else:
            LAYOUTS = {
                'joint': {'shape': (8, 3), 'size': 8*3},
                'parent': {'shape': (8, 3), 'size': 8*3},
            }
            if self.use_conf_jp:
                LAYOUTS['conf_j'] = {'shape': (8, 1), 'size': 8}
                LAYOUTS['conf_p'] = {'shape': (8, 1), 'size': 8}
            if self.use_conf_skin:
                LAYOUTS['conf_skin'] = {'shape': (8, 1), 'size': 8}
            # Define skin features at the end
            if self.predict_skin:
                LAYOUTS['skin_feat'] = {'shape': (8, self.skin_feat_channels), 'size': 8*self.skin_feat_channels}
                self.skin_feat_channels = 8 * self.skin_feat_channels
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start - (self.skin_feat_channels if self.predict_skin else 0)
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a skeleton based on the specified sparse voxel structures.
        Args:
            cubefeats [SparseTensor] : Sparse Tensor attrs about cube weights
        Returns:
            return s a dictionary with joints, parents, skin features, and positions.
        """
        coords = cubefeats.coords[:, 1:]
        joints, parents, skin_feats, conf_j, conf_p, conf_skin = [self.get_layout(cubefeats.feats, name) for name in ['joint', 'parent', 'skin_feat', 'conf_j', 'conf_p', 'conf_skin']]
        if conf_skin is not None:
            conf_skin = torch.sigmoid(conf_skin)
        if self.defined_on_center:
            positions = (coords + 0.5) / self.res - 0.5
            if self.jp_hyper_continuous:
                conf_j = torch.sigmoid(conf_j)
                conf_p = torch.sigmoid(conf_p)
                conf_skin = conf_j
            if self.jp_residual_fields:
                joints = joints + positions
                parents = parents + positions
            results = {
                'joints': joints,
                'parents': parents,
                'skin_feats': skin_feats,
                'positions': positions,
                'reg_loss': 0,  # No reg loss for skeleton extraction
                'conf_j': conf_j,
                'conf_p': conf_p,
                'conf_skin': conf_skin,
                'skin_pred': None, 
                'skin_feats_joints_var_loss': None,
                'jp_hyper_continuous': self.jp_hyper_continuous,
                'jp_residual_fields': self.jp_residual_fields,
                'joints_grouped': None,
                'parents_grouped': None,
            }
        else:
            results = {}
            skin_feat_channels = self.skin_feat_channels // 8 if self.predict_skin else 0
            v_attrs = [joints, parents]
            if self.predict_skin:
                v_attrs.append(skin_feats)
            if self.use_conf_jp:
                v_attrs.append(conf_j)
                v_attrs.append(conf_p)
            if self.use_conf_skin:
                v_attrs.append(conf_skin)
            v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
            positions = ((v_pos + 0.5) / self.res - 0.5)
            joints_grid, parents_grid = v_attrs[:, :3], v_attrs[:, 3:6]
            skin_feats_grid, conf_j_grid, conf_p_grid, conf_skin_grid = None, None, None, None
            if self.predict_skin:
                skin_feats_grid = v_attrs[:, 6:6+skin_feat_channels]
            if self.use_conf_jp:
                conf_j_grid = v_attrs[:, 6+skin_feat_channels:7+skin_feat_channels]
                conf_p_grid = v_attrs[:, 7+skin_feat_channels:8+skin_feat_channels]
            if self.use_conf_skin:
                conf_skin_grid = v_attrs[:, 8+skin_feat_channels:9+skin_feat_channels] if self.use_conf_jp else v_attrs[:, 6+skin_feat_channels:7+skin_feat_channels]
            if self.jp_hyper_continuous:
                conf_j_grid = torch.sigmoid(conf_j_grid)
                conf_p_grid = torch.sigmoid(conf_p_grid)
                conf_skin_grid = conf_j_grid
            if self.jp_residual_fields:
                joints_grid = joints_grid + positions
                parents_grid = parents_grid + positions
            results.update({
                'joints': joints_grid,
                'parents': parents_grid,
                'skin_feats': skin_feats_grid,
                'positions': positions,
                'reg_loss': reg_loss if training else 0,
                'conf_j': conf_j_grid,
                'conf_p': conf_p_grid,
                'conf_skin': conf_skin_grid,
                'skin_pred': None, 
                'skin_feats_joints_var_loss': None,
                'jp_hyper_continuous': self.jp_hyper_continuous,
                'jp_residual_fields': self.jp_residual_fields,
                'joints_grouped': None,
                'parents_grouped': None,
            })
        return edict(results)
