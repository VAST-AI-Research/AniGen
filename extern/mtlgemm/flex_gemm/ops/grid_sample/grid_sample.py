from typing import *
import torch
from torch.autograd import Function
from .. import grid_sample, utils
from ... import kernels


class GridSample3dFunction(Function):
    
    @staticmethod
    def _nearest_fwd(
        ctx,
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
        indices = kernels.cuda.hashmap_build_grid_sample_3d_nearest_neighbor_map(
            hashmap_keys, hashmap_vals,
            coords.int(),
            query_pts,
            W, H, D
        ).int()
        valid = (indices != 0xffffffff)
        indices.clamp_min_(0)
        out = valid.unsqueeze(-1) * feats.index_select(0, indices.reshape(-1)).reshape(B, L, C)
                
        ctx.save_for_backward(indices, valid)
        ctx.N = N
        ctx.C = C
        
        return out
    
    @staticmethod
    def _nearest_bwd(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        indices, valid = ctx.saved_tensors
        
        grad_feats = torch.zeros(
            (ctx.N, ctx.C),
            device=grad_output.device,
            dtype=grad_output.dtype
        )
        
        grad_feats.index_add_(
            0,
            indices[valid],
            grad_output[valid].reshape(-1, ctx.C)
        )
        return grad_feats, None, None, None, None
    
    @staticmethod
    def _trilinear_fwd(
        ctx,
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
        indices, weight = kernels.cuda.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
            hashmap_keys, hashmap_vals,
            coords.int(),
            query_pts,
            W, H, D
        )
        
        out = kernels.triton.indice_weighed_sum_fwd(
            feats,
            indices.view(-1, 8),
            weight.view(-1, 8),
        ).view(B, L, C)
        
        ctx.save_for_backward(indices, weight)
        ctx.N = N
        ctx.C = C
        
        return out
    
    @staticmethod
    def _trilinear_bwd(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        indices, weight = ctx.saved_tensors

        grad_feats = torch.zeros(
            (ctx.N, ctx.C),
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        grad_feats = kernels.triton.indice_weighed_sum_bwd_input(
            grad_output.reshape(-1, ctx.C).contiguous(),
            indices.view(-1, 8),
            weight.view(-1, 8),
            ctx.N,
        ).view(ctx.N, ctx.C)

        return grad_feats, None, None, None, None
    
    @staticmethod
    def forward(
        ctx,
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
            coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
            shape (torch.Size): The spatial shape of the sparse tensor
            grid (torch.Tensor): A [B, L, 3] tensor containing the query points
            mode (str): The interpolation mode to use (nearest, trilinear)
        
        Returns:
            torch.Tensor: A [B, L, C] tensor containing the sampled features
        """
        assert mode in ["nearest", "trilinear"], "Invalid interpolation mode"
        
        ctx.mode = mode
        
        if mode == "nearest":
            return GridSample3dFunction._nearest_fwd(ctx, feats, coords, shape, grid)
        else:
            return GridSample3dFunction._trilinear_fwd(ctx, feats, coords, shape, grid)
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        if ctx.needs_input_grad[0]:
            if ctx.mode == "nearest":
                return GridSample3dFunction._nearest_bwd(ctx, grad_output)
            else:
                return GridSample3dFunction._trilinear_bwd(ctx, grad_output)
        else:
            return None, None, None, None, None


def grid_sample_3d(
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
        coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
        shape (torch.Size): The spatial shape of the sparse tensor
        grid (torch.Tensor): A [B, L, 3] tensor containing the query points
        mode (str): The interpolation mode to use (nearest, trilinear)
    
    Returns:
        torch.Tensor: A [B, L, C] tensor containing the sampled features
    """
    return GridSample3dFunction.apply(feats, coords, shape, grid, mode)
