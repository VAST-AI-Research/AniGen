from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.jit
def sparse_submanifold_conv_fwd_implicit_gemm_kernel(
    input,
    weight,
    bias,
    neighbor,
    output,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for N dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Sparse submanifold convolution forward kernel using implicit GEMM.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (N, V)
        output (pointer): A pointer to the output tensor of shape (N, Co)
    """
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_n = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    offset_n = (block_id_n * B1 + tl.arange(0, B1)) % N         # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co      # (B2,)
    offset_k = tl.arange(0, BK)                                 # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Calculate pointers to weight matrix.
    weight_ptr = weight + (offset_co[None, :] * V * Ci + offset_k[:, None])     # (BK, B2)
    
    # Iterate along V*Ci dimension.
    for k in range(num_k * V):
        v = k // num_k
        bk = k % num_k
        # Calculate pointers to input matrix.
        neighbor_offset_n = tl.load(neighbor + offset_n * V + v)                                # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset_n[:, None].to(tl.int64) * Ci + offset_k[None, :])     # (B1, BK)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset_n != 0xffffffff
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
        # Advance the pointers to the next Ci block.
        weight_ptr += min(BK, Ci - bk * BK)
    c = accumulator.to(input.type.element_ty)
            
    # add bias
    if bias is not None:
        bias_block = tl.load(bias + offset_co)
        c += bias_block[None, :]
                
    # Write back the block of the output matrix with masks.
    out_offset_n = block_id_n * B1 + tl.arange(0, B1)
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + (out_offset_n[:, None] * Co + out_offset_co[None, :])
    out_mask = (out_offset_n[:, None] < N) & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, c, mask=out_mask)


def sparse_submanifold_conv_fwd_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, Ci, Co, V = neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    # Allocate output matrix output.
    output = torch.empty((N, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(N, META['B1']),)
    sparse_submanifold_conv_fwd_implicit_gemm_kernel[grid](
        input, weight, bias, neighbor, output,
        N, LOGN, Ci, Co, V,
        allow_tf32=config.allow_tf32,
    )
    return output
