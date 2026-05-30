from typing import *
import math
import torch
import triton
import triton.language as tl
from ....utils.autotuner import triton_autotune
from . import config


heuristics_bwd_input = {
    'valid_kernel': lambda args: args['valid_kernel'](args['B1']),
    'valid_kernel_seg': lambda args: args['valid_kernel_seg'](args['B1']),
}


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.heuristics(heuristics_bwd_input)
@triton.jit
def sparse_submanifold_conv_bwd_input_masked_implicit_gemm_kernel(
    grad_output,
    weight,
    neighbor,
    sorted_idx,
    grad_input,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for N dimension
    B2: tl.constexpr,   # Block size for Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Co)
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Huristic parameters
    valid_kernel,
    valid_kernel_seg,
):
    """
    Sparse submanifold convolution backward to input kernel using masked implicit GEMM.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (N, Co)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        neighbor (pointer): A pointer to the neighbor tensor of shape (N, V)
        sorted_idx (pointer): A pointer to the sorted index tensor of shape (N)
        valid_kernel (pointer): A pointer to the valid neighbor index tensor of shape (L)
        valid_kernel_seg (pointer): A pointer to the valid neighbor index segment tensor of shape (BLOCK_N + 1)
        grad_input (pointer): A pointer to the gradient of the input tensor of shape (N, Ci)
    """
    block_id = tl.program_id(axis=0)
    block_dim_ci = tl.cdiv(Ci, B2)
    block_id_ci = block_id % block_dim_ci
    block_id_n = block_id // block_dim_ci
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Co, BK)  # Number of blocks in K dimension
    valid_kernel_start = tl.load(valid_kernel_seg + block_id_n)
    valid_kernel_seglen = tl.load(valid_kernel_seg + block_id_n + 1) - valid_kernel_start
    offset_n = block_id_n * B1 + tl.arange(0, B1)
    n_mask = offset_n < N
    offset_sorted_n = tl.load(sorted_idx + offset_n, mask=n_mask, other=0)  # (B1,)
    offset_ci = (block_id_ci * B2 + tl.arange(0, B2)) % Ci      # (B2,)
    offset_k = tl.arange(0, BK)                                 # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)    
    
    # Iterate along V*Co dimension.
    for k in range(num_k * valid_kernel_seglen):
        v = k // num_k
        bk = k % num_k
        v = tl.load(valid_kernel + valid_kernel_start + v)
        # Calculate pointers to grad_output matrix.
        neighbor_offset_n = tl.load(neighbor + offset_sorted_n * V + v)                                     # (B1,)
        grad_output_ptr = grad_output + bk * BK + (neighbor_offset_n[:, None].to(tl.int64) * Co + offset_k[None, :])     # (B1, BK)
        # Calculate pointers to weight matrix.
        weight_ptr = weight + (((offset_k[:, None] + bk * BK) * V + V - 1 - v) * Ci + offset_ci[None, :])   # (BK, B2)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset_n != 0xffffffff
        k_mask = offset_k < Co - bk * BK
        grad_output_block = tl.load(grad_output_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                              # (B1, B2)
    c = accumulator.to(grad_output.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    grad_input_offset_n = offset_sorted_n
    grad_input_offset_ci = block_id_ci * B2 + tl.arange(0, B2)
    grad_input_ptr = grad_input + (grad_input_offset_n[:, None] * Ci + grad_input_offset_ci[None, :])
    grad_input_mask = n_mask[:, None] & (grad_input_offset_ci[None, :] < Ci)
    tl.store(grad_input_ptr, c, mask=grad_input_mask)

    
@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.jit
def sparse_submanifold_conv_bwd_weight_masked_implicit_gemm_kernel(
    grad_output,
    input,
    valid_signal_i,
    valid_signal_o,
    valid_signal_seg,
    grad_weight,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for Co dimension
    B2: tl.constexpr,   # Block size for Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (N)
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Sparse submanifold convolution backward to weight kernel using implicit GEMM.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (N, Co)
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        valid_signal_i (pointer): A pointer to the valid input signal tensor of shape (L)
        valid_signal_o (pointer): A pointer to the valid output signal tensor of shape (L)
        valid_signal_seg (pointer): A pointer to the valid signal index segment tensor of shape (V + 1)
        grad_weight (pointer): A pointer to the gradient of the weight tensor of shape (Co, V, Ci)
    """
    num_blocks_co = tl.cdiv(Co, B1)
    num_blocks_ci = tl.cdiv(Ci, B2)
    block_id = tl.program_id(axis=0)
    block_id_co = block_id % num_blocks_co
    block_id_ci = block_id // num_blocks_co % num_blocks_ci
    block_id_v = block_id // (num_blocks_co * num_blocks_ci)
    
    # Create pointers for submatrices of A and B.
    valid_signal_start = tl.load(valid_signal_seg + block_id_v)
    valid_signal_seglen = tl.load(valid_signal_seg + block_id_v + 1) - valid_signal_start
    num_k = tl.cdiv(valid_signal_seglen, BK)  # Number of blocks in K dimension
    offset_co = (block_id_co * B1 + tl.arange(0, B1)) % Co                          # (B1,)
    offset_ci = (block_id_ci * B2 + tl.arange(0, B2)) % Ci                          # (B2,)
    offset_k = tl.arange(0, BK)                                                     # (BK,)
    
    valid_signal_i_ptr = valid_signal_i + valid_signal_start + offset_k
    valid_signal_o_ptr = valid_signal_o + valid_signal_start + offset_k
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)   
    
    # Iterate along V*Ci dimension.
    for k in range(num_k):
        # Calculate pointers to input and grad_output matrix.
        mask = offset_k < valid_signal_seglen - k * BK
        input_offset_n = tl.load(valid_signal_i_ptr, mask=mask, other=0)                            # (BK,)
        grad_output_offset_n = tl.load(valid_signal_o_ptr, mask=mask, other=0)                      # (BK,)
        input_ptr = input + (input_offset_n[:, None].to(tl.int64) * Ci + offset_ci[None, :])                     # (BK, B2)
        grad_output_ptr = grad_output + grad_output_offset_n[None, :].to(tl.int64) * Co + offset_co[:, None]     # (B1, BK)
        # Load the next block of input and grad_output.
        input_block = tl.load(input_ptr, mask=mask[:, None], other=0.0)
        grad_output_block = tl.load(grad_output_ptr, mask=mask[None, :], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, input_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)
        # Advance pointers.
        valid_signal_i_ptr += BK
        valid_signal_o_ptr += BK
    c = accumulator.to(grad_output.type.element_ty)
                
    # Write back the block of the output matrix with masks.
    grad_weight_offset_co = block_id_co * B1 + tl.arange(0, B1)
    grad_weight_offset_ci = block_id_ci * B2 + tl.arange(0, B2)
    grad_weight_ptr = grad_weight + (grad_weight_offset_co[:, None] * V * Ci + block_id_v * Ci + grad_weight_offset_ci[None, :])
    grad_weight_mask = (grad_weight_offset_co[:, None] < Co) & (grad_weight_offset_ci[None, :] < Ci)
    tl.store(grad_weight_ptr, c, mask=grad_weight_mask)
    

def sparse_submanifold_conv_bwd_masked_implicit_gemm(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
    sorted_idx: torch.Tensor,
    valid_kernel: Callable[[int], torch.Tensor],
    valid_kernel_seg: Callable[[int], torch.Tensor],
    valid_signal_i: torch.Tensor,
    valid_signal_o: torch.Tensor,
    valid_signal_seg: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert grad_output.is_contiguous(), "Matrix grad_output must be contiguous"
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert neighbor.is_contiguous(), "Matrix neighbor must be contiguous"
    N, Ci, Co, V = neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    
    grad_input, grad_weight, grad_bias = None, None, None
    
    # Grad for input
    if input.requires_grad:
        # Allocate output matrix output.
        grad_input = torch.empty((N, Ci), device=input.device, dtype=input.dtype)
        # Launch the kernel.
        grid = lambda META: (triton.cdiv(Ci, META['B2']) * triton.cdiv(N, META['B1']),)
        sparse_submanifold_conv_bwd_input_masked_implicit_gemm_kernel[grid](
            grad_output,
            weight,
            neighbor,
            sorted_idx,
            grad_input,
            N, LOGN, Ci, Co, V,
            valid_kernel=valid_kernel,
            valid_kernel_seg=valid_kernel_seg,
            allow_tf32=config.allow_tf32,
        )
        
    # Grad for weight
    if weight.requires_grad:
        # Allocate output matrix output.
        grad_weight = torch.empty((Co, V, Ci), device=weight.device, dtype=weight.dtype)
        # Launch the kernel.
        grid = lambda META: (triton.cdiv(Co, META['B1']) * triton.cdiv(Ci, META['B2']) * V,)
        sparse_submanifold_conv_bwd_weight_masked_implicit_gemm_kernel[grid](
            grad_output,
            input,
            valid_signal_i,
            valid_signal_o,
            valid_signal_seg,
            grad_weight,
            N, LOGN, Ci, Co, V,
            allow_tf32=config.allow_tf32,
        )
        
    # Grad for bias
    if bias is not None and bias.requires_grad:
        grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias
