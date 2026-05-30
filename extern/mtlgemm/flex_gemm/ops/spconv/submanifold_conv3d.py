from typing import *
import platform
import torch
from torch.autograd import Function
from . import Algorithm
from .. import spconv, utils
from ... import kernels


class SubMConv3dNeighborCache:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def compute_kernel_idx(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg
        
    def valid_kernel_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_{block_size}']
    
    def valid_kernel_seg_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_seg_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_seg_{block_size}']


class SubMConv3dFunction(Function):
    @staticmethod
    def _compute_neighbor_cache(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int]
    ) -> SubMConv3dNeighborCache:
        assert coords.is_contiguous(), "Coords should be contiguous"
        assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
        N, C, W, H, D = shape
        
        hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(spconv.HASHMAP_RATIO * coords.shape[0]), coords.device)

        if spconv.ALGORITHM in [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda or (platform.system() == 'Darwin'):
                neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
                    hashmap_keys, hashmap_vals, coords,
                    W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            return SubMConv3dNeighborCache(**{
                'neighbor_map': neighbor_map,
            })
        
        elif spconv.ALGORITHM in [Algorithm.MASKED_IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda or (platform.system() == 'Darwin'):
                neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
                    hashmap_keys, hashmap_vals, coords,
                    W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            V = kernel_size[0] * kernel_size[1] * kernel_size[2]
            assert V <= 32, "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"
            
            gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = \
                kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
            
            return SubMConv3dNeighborCache(**{
                'neighbor_map': neighbor_map,
                'gray_code': gray_code,
                'sorted_idx': sorted_idx,
                'valid_signal_seg': valid_signal_seg,
                'valid_signal_i': valid_signal_i,
                'valid_signal_o': valid_signal_o,
            })
                
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")

    def _compute_neighbor_cache_torch(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int]
    ) -> SubMConv3dNeighborCache:
        assert spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM, "Only explicit_gemm is supported for torch implementation"
        N, C, W, H, D = shape
        L = coords.shape[0]
        assert N * W * H * D <= 2**32, "Currently, the max number of elements in a tensor is 2^32"
        M = torch.tensor([W * H * D, H * D, D, 1], device=coords.device).int()
        
        keys = (coords * M[None]).sum(dim=-1)
        sorted_keys, indices = torch.sort(keys)
        
        # Compute neighbor coords
        offset = torch.meshgrid(
            torch.arange(-(kernel_size[0] // 2) * dilation[0], kernel_size[0] // 2 * dilation[0] + 1, dilation[0]),
            torch.arange(-(kernel_size[1] // 2) * dilation[1], kernel_size[1] // 2 * dilation[1] + 1, dilation[1]),
            torch.arange(-(kernel_size[2] // 2) * dilation[2], kernel_size[2] // 2 * dilation[2] + 1, dilation[2]),
            indexing='ij'
        )
        offset = torch.stack(offset, dim=-1).reshape(-1, 3).int().to(coords.device)
        neighbor_coords = coords.unsqueeze(1).repeat(1, kernel_size[0] * kernel_size[1] * kernel_size[2], 1)
        neighbor_coords[:, :, -3:] += offset.unsqueeze(0)                                    # [N, kernel_vol, 4]
        neighbor_coords = neighbor_coords.reshape(-1, 4)                                    # [N * kernel_vol, 4]
        neighbor_valid = (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < W) & \
                         (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < H) & \
                         (neighbor_coords[:, 3] >= 0) & (neighbor_coords[:, 3] < D)
        neighbor_keys = (neighbor_coords * M[None]).sum(dim=-1)
        neighbor_search_indices = torch.searchsorted(sorted_keys, neighbor_keys)
        neighbor_search_indices = torch.clamp(neighbor_search_indices, 0, sorted_keys.shape[0] - 1)
        neighbor_valid &= sorted_keys[neighbor_search_indices] == neighbor_keys
        neighbor_indices = torch.full((L * kernel_size[0] * kernel_size[1] * kernel_size[2],), 0xffffffff, dtype=torch.long, device=coords.device)
        neighbor_indices[neighbor_valid] = indices[neighbor_search_indices[neighbor_valid]]
        return SubMConv3dNeighborCache(**{'neighbor_map': neighbor_indices.reshape(L, -1).to(torch.uint32)})
        
    @staticmethod
    def _sparse_submanifold_conv_forward(
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert feats.is_contiguous(), "Input features should be contiguous"
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw
        
        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:        
            neighbor_map = neighbor_cache['neighbor_map']
            
            # im2col
            im2col = torch.zeros((N * V, Ci), device=feats.device, dtype=feats.dtype)
            mask = neighbor_map.view(-1) != 0xffffffff
            im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Ci)
            
            # addmm
            weight = weight.view(Co, V * Ci).transpose(0, 1)
            if bias is not None:
                output = torch.addmm(bias, im2col, weight)
            else:
                output = torch.mm(im2col, weight)
        
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return output

    @staticmethod
    def _sparse_submanifold_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw

        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache['neighbor_map']
            
            if feats.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Co), device=feats.device, dtype=feats.dtype)
                if platform.system() == 'Darwin':
                    inv_neighbor_map = torch.flip(neighbor_map.to(torch.int32), [1]).to(torch.uint32)
                else:
                    inv_neighbor_map = torch.flip(neighbor_map, [1])
                mask = inv_neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Co)
                
                # addmm
                grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
            else:
                grad_input = None
                
            if weight.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
                mask = neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Ci)
                
                # addmm
                grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous().view(Co, Kw, Kh, Kd, Ci)
            else:
                grad_weight = None
            
            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
        
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return grad_input, grad_weight, grad_bias
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        neighbor_cache: Optional[SubMConv3dNeighborCache],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        dilation: Tuple[int, int, int] = (1, 1, 1),
    ) -> Tuple[torch.Tensor, SubMConv3dNeighborCache]:
        Co, Kw, Kh, Kd, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"
        
        # check if neighbor map is already computed
        if neighbor_cache is None:
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), dilation)
            
        # compute output
        output = SubMConv3dFunction._sparse_submanifold_conv_forward(feats, neighbor_cache, weight, bias)
        
        # save for backward
        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        
        return output, neighbor_cache
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        
        grad_input, grad_weight, grad_bias = SubMConv3dFunction._sparse_submanifold_conv_backward(grad_output, feats, neighbor_cache, weight, bias)
        
        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None


def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SubMConv3dNeighborCache] = None,
    dilation: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[torch.Tensor, SubMConv3dNeighborCache]:
    """
    Sparse submanifold convolution for 3D input.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[torch.Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConv3dNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        dilation (Tuple[int, int, int]): dilation rate.

    Returns:
        Tuple[torch.Tensor, SubMConv3dNeighborCache]:
            - output (torch.Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConv3dNeighborCache): neighbor cache for backward.
    """
    return SubMConv3dFunction.apply(feats, coords, shape, neighbor_cache, weight, bias, dilation)
