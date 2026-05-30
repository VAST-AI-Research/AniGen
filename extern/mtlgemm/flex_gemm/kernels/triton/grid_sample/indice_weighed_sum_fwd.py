from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'M', 'C', 'V']
)
@triton.jit
def indice_weighed_sum_fwd_kernel(
    input,
    indices,
    weight,
    output,
    # Tensor dimensions
    LOGN, M, C, V: tl.constexpr,
    # Meta-parameters
    BM: tl.constexpr,   # Block size for N dimension
    BK: tl.constexpr,   # Block size for C dimension
):
    """
    Forward pass of the weighted sum of the input features using the indices.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, C)
        indices (pointer): A pointer to the indices tensor of shape (M, V)
        weight (pointer): A pointer to the weight tensor of shape (M, V)
        output (pointer): A pointer to the output tensor of shape (M, C)
    """
    block_id = tl.program_id(axis=0)
    num_k = tl.cdiv(C, BK)  # Number of blocks in K dimension
    block_id_m = block_id // num_k  # Block ID in M dimension
    block_id_k = block_id % num_k   # Block ID in K dimension
    
    offset_m = (block_id_m * BM + tl.arange(0, BM)) % M           # (BM,)
    offset_k = (block_id_k * BK + tl.arange(0, BK)) % C           # (BK,)
    
    # Create a block of the output matrix.
    accumulator = tl.zeros((BM, BK), dtype=tl.float32)          # (BM, BK)
        
    # Iterate along V*C dimension.
    for v in range(V):
        # Calculate pointers
        neigh_idx = tl.load(indices + offset_m * V + v)                         # (BM,)
        input_ptr = input + (neigh_idx[:, None] * C + offset_k[None, :])        # (BM, BK)
        weight_ptr = weight + offset_m * V + v                                          # (BM,)
        # Load the next block of input and weight.
        neigh_mask = neigh_idx != 0xffffffff
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None], other=0.0)
        weight_block = tl.load(weight_ptr)
        # Accumulate along the K dimension.
        accumulator += input_block * weight_block[:, None]
    c = accumulator.to(input.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    out_ptr = output + (offset_m[:, None] * C + offset_k[None, :])
    out_mask = (offset_m[:, None] < M) & (offset_k[None, :] < C)
    tl.store(out_ptr, c, mask=out_mask)


def indice_weighed_sum_fwd(
    input: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert indices.is_contiguous(), "Matrix indices must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert indices.shape == weight.shape, "Indices and weight must have the same shape"
    N, M, C, V = input.shape[0], indices.shape[0], input.shape[1], weight.shape[1]
    LOGN = int(math.log2(N))
    # Allocate output matrix output.
    output = torch.empty((M, C), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(C, META['BK']) * triton.cdiv(M, META['BM']),)
    indice_weighed_sum_fwd_kernel[grid](
        input, indices, weight, output,
        LOGN, M, C, V,
    )
    return output
