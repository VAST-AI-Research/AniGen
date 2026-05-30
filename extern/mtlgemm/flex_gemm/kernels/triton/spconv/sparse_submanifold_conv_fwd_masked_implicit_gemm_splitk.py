from typing import *
import math
import torch
import triton
import triton.language as tl
from ..utils import get_num_sm
from ....utils.autotuner import triton_autotune, autotune
from . import config
from .sparse_submanifold_conv_fwd_masked_implicit_gemm import sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel


heuristics = {
    'valid_kernel': lambda args: args['valid_kernel'](args['B1']),
    'valid_kernel_seg': lambda args: args['valid_kernel_seg'](args['B1']),
}


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V', 'SPLITK', 'allow_tf32'],
)
@triton.heuristics(heuristics)
@triton.jit
def sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel(
    input,
    weight,
    bias,
    neighbor,
    sorted_idx,
    output,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for N dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    SPLITK: tl.constexpr,  # Split K dimension into multiple sub-dimensions
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Huristic parameters
    valid_kernel,
    valid_kernel_seg,
):
    """
    Sparse submanifold convolution forward kernel using masked implicit GEMM split-k.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (N, V)
        sorted_idx (pointer): A pointer to the sorted index tensor of shape (N)
        valid_kernel (pointer): A pointer to the valid neighbor index tensor of shape (L)
        valid_kernel_seg (pointer): A pointer to the valid neighbor index segment tensor of shape (BLOCK_N + 1)
        output (pointer): A pointer to the output tensor of shape (N, Co)
    """
    block_id_k = tl.program_id(axis=1)  # SplitK dimension
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_n = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    valid_kernel_start = tl.load(valid_kernel_seg + block_id_n)
    valid_kernel_seglen = tl.load(valid_kernel_seg + block_id_n + 1) - valid_kernel_start
    k_start = tl.cdiv(num_k * valid_kernel_seglen * block_id_k, SPLITK)
    k_end = tl.cdiv(num_k * valid_kernel_seglen * (block_id_k + 1), SPLITK)
    offset_n = block_id_n * B1 + tl.arange(0, B1)
    n_mask = offset_n < N
    offset_sorted_n = tl.load(sorted_idx + offset_n, mask=n_mask, other=0)  # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co                  # (B2,)
    offset_k = tl.arange(0, BK)                                             # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Iterate along V*Ci dimension.
    for k in range(k_start, k_end):
        v = k // num_k
        bk = k % num_k
        v = tl.load(valid_kernel + valid_kernel_start + v)
        # Calculate pointers to input matrix.
        neighbor_offset_n = tl.load(neighbor + offset_sorted_n * V + v)                             # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset_n[:, None].to(tl.int64) * Ci + offset_k[None, :])         # (B1, BK)
        # Calculate pointers to weight matrix.
        weight_ptr = weight + v * Ci + bk * BK + (offset_co[None, :] * V * Ci + offset_k[:, None])  # (BK, B2)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset_n != 0xffffffff
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)
            
    # add bias
    if bias is not None and block_id_k == 0:
        bias_block = tl.load(bias + offset_co)
        accumulator += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_n = offset_sorted_n
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + block_id_k * N * Co + (out_offset_n[:, None] * Co + out_offset_co[None, :])
    out_mask = n_mask[:, None] & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, accumulator, mask=out_mask)


def sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_configs(input, weight, bias, neighbor, sorted_idx, valid_kernel, valid_kernel_seg):
    N, Co = neighbor.shape[0], weight.shape[0]
    MAX_NB1 = (N + 128 - 1) // 128
    MAX_NB2 = (Co + 128 - 1) // 128
    NUM_BLOCKS = MAX_NB1 * MAX_NB2
    MIN_NUM_BLOCKS = get_num_sm()
    MAX_NUM_BLOCKS = 32 * get_num_sm()
    MIN_NUM_BLOCKS_LOG2 = max(0, int(math.log2(MIN_NUM_BLOCKS / NUM_BLOCKS)))
    MAX_NUM_BLOCKS_LOG2 = max(1, int(math.log2(MAX_NUM_BLOCKS / NUM_BLOCKS) + 1))
    configs = []
    for i in range(MIN_NUM_BLOCKS_LOG2, MAX_NUM_BLOCKS_LOG2):
        configs.append({'SPLITK': 2 ** i})
    return configs


def sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_keys(input, weight, bias, neighbor, sorted_idx, valid_kernel, valid_kernel_seg):
    N, Ci, Co, V = neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    return f'(2^{int(math.log2(N))}, {Ci}, {Co}, {V})'


@autotune(
    config_fn=sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_configs,
    key_fn=sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_keys,
)
def sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
    sorted_idx: torch.Tensor,
    valid_kernel: Callable[[int], torch.Tensor],
    valid_kernel_seg: Callable[[int], torch.Tensor],
    SPLITK: int = 1,
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, Ci, Co, V = neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    # Launch the kernel.
    if SPLITK == 1:
        output = torch.empty((N, Co), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(N, META['B1']),)
        sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel[grid](
            input, weight, bias, neighbor, sorted_idx, output,
            N, LOGN, Ci, Co, V,  #
            valid_kernel=valid_kernel,
            valid_kernel_seg=valid_kernel_seg,
            allow_tf32=config.allow_tf32,
        )
        return output
    else:
        output = torch.empty((SPLITK, N, Co), device=input.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(N, META['B1']), SPLITK)
        sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel[grid](
            input, weight, bias, neighbor, sorted_idx, output,
            N, LOGN, Ci, Co, V,  #
            valid_kernel=valid_kernel,
            valid_kernel_seg=valid_kernel_seg,
            SPLITK=SPLITK,
            allow_tf32=config.allow_tf32,
        )
        return output.sum(dim=0).to(input.dtype)
