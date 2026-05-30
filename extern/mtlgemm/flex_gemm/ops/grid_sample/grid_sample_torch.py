from typing import *
import torch
from .. import grid_sample, utils
from ... import kernels


class GridSample3dTorch:
    
    @staticmethod
    def _nearest(
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        query_pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples the input sparse tensor at the given points using nearest neighbor interpolation.
        
        Args:
            feats (torch.Tensor): A [N, C] tensor containing the features to sample from
            coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
            shape (torch.Size): The spatial shape of the sparse tensor
            query_pts (torch.Tensor): A [B, L, 3] tensor containing the query points
        
        Returns:
            torch.Tensor: A [B, L, C] tensor containing the sampled features
        """
        assert feats.dim() == 2, f"Features must be of shape [N, C], got {feats.shape}"
        assert coords.dim() == 2 and coords.shape[1] == 4, f"Coords must be of shape [N, 4], got {coords.shape}"
        assert query_pts.dim() == 3 and query_pts.shape[2] == 3, f"Query points must be of shape [B, L, 3], got {query_pts.shape}"
        assert feats.shape[0] == coords.shape[0], "Number of features must match number of coordinates"
        
        N = coords.shape[0]
        B, L = query_pts.shape[:2]
        C, W, H, D = shape[-4:]
        
        hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(grid_sample.HASHMAP_RATIO * coords.shape[0]), coords.device)
        kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(
            hashmap_keys, hashmap_vals,
            coords.int(),
            W, H, D
        )
        
        neigh_pts = torch.cat([
            torch.arange(B, dtype=torch.int32, device=query_pts.device).reshape(B, 1, 1).repeat(1, L, 1),
            query_pts.int()
        ], dim=2)                       # [B, L, 4]
                
        idx = kernels.cuda.hashmap_lookup_3d_cuda(
            hashmap_keys, hashmap_vals,
            neigh_pts.view(-1, 4),
            W, H, D
        ).int()
        valid = (idx != 0xffffffff)
        feat = valid.unsqueeze(1) * feats.index_select(0, idx.clamp_min(0))
        
        return feat.view(B, L, C)
    
    @staticmethod
    def _trilinear(
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        query_pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples the input sparse tensor at the given points using trilinear interpolation.
        
        Args:
            feats (torch.Tensor): A [N, C] tensor containing the features to sample from
            coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
            shape (torch.Size): The spatial shape of the sparse tensor
            query_pts (torch.Tensor): A [B, L, 3] tensor containing the query points
        
        Returns:
            torch.Tensor: A [B, L, C] tensor containing the sampled features
        """
        assert feats.dim() == 2, f"Features must be of shape [N, C], got {feats.shape}"
        assert coords.dim() == 2 and coords.shape[1] == 4, f"Coords must be of shape [N, 4], got {coords.shape}"
        assert query_pts.dim() == 3 and query_pts.shape[2] == 3, f"Query points must be of shape [B, L, 3], got {query_pts.shape}"
        assert feats.shape[0] == coords.shape[0], "Number of features must match number of coordinates"
        
        N = coords.shape[0]
        B, L = query_pts.shape[:2]
        C, W, H, D = shape[-4:]
        
        hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(grid_sample.HASHMAP_RATIO * coords.shape[0]), coords.device)
        kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(
            hashmap_keys, hashmap_vals,
            coords.int(),
            W, H, D
        )
        
        neigh_pts = torch.cat([
            torch.arange(B, dtype=torch.int32, device=query_pts.device).reshape(B, 1, 1).repeat(1, 8 * L, 1),
            (query_pts.unsqueeze(2) + torch.tensor([
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5,  0.5], 
                [-0.5,  0.5, -0.5], 
                [-0.5,  0.5,  0.5], 
                [ 0.5, -0.5, -0.5], 
                [ 0.5, -0.5,  0.5], 
                [ 0.5,  0.5, -0.5], 
                [ 0.5,  0.5,  0.5],
            ], device=query_pts.device).unsqueeze(0)).reshape(B, 8 * L, 3).int()
        ], dim=2)                                   # [B, 8*L, 4]
        
        idx = kernels.cuda.hashmap_lookup_3d_cuda(
            hashmap_keys, hashmap_vals,
            neigh_pts.view(-1, 4),
            W, H, D
        ).int()
        valid = (idx != 0xffffffff)
        weight = torch.where(
            valid,
            torch.prod(1 - torch.abs(neigh_pts[..., 1:] + 0.5 - query_pts.repeat_interleave(8, dim=1)), dim=-1).reshape(-1),
            torch.zeros(valid.shape[0], dtype=torch.float32, device=query_pts.device)
        )                                                                                                   # [M*8]
        feat = weight.unsqueeze(1) * feats.index_select(0, idx.clamp_min(0))                                # [M*8, C]
        weight = weight.reshape(-1, 8).sum(dim=1)                                                           # [M]
        feat = feat.reshape(-1, 8, C).sum(dim=1) / torch.clamp_min(weight.unsqueeze(1), 1e-12)              # [M, C]
        
        return feat.view(B, L, C)
    
    @staticmethod
    def forward(
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        grid: torch.Tensor,
        mode: str = "trilinear",
    ) -> torch.Tensor:
        """
        Samples the input sparse tensor at the given points using the specified interpolation mode.
        
        Args:
            feats (torch.Tensor): A [N, C] tensor containing the features to sample from
            coords (torch.Tensor): A [N, ..., 4] tensor containing the coordinates of the features
            shape (torch.Size): The spatial shape of the sparse tensor
            grid (torch.Tensor): A [B, L, 3] tensor containing the query points
            mode (str): The interpolation mode to use (nearest, trilinear)
        
        Returns:
            torch.Tensor: A [B, L, C] tensor containing the sampled features
        """
        assert mode in ["nearest", "trilinear"], "Invalid interpolation mode"
        
        if mode == "nearest":
            return GridSample3dTorch._nearest(feats, coords, shape, grid)
        else:
            return GridSample3dTorch._trilinear(feats, coords, shape, grid)
        

grid_sample_3d_torch = GridSample3dTorch.forward
