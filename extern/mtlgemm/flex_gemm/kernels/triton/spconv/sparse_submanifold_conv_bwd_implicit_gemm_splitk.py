from typing import *
import math
import torch
import triton
import triton.language as tl
from ..utils import get_num_sm
from ....utils.autotuner import triton_autotune, autotune
from . import config
from .sparse_submanifold_conv_bwd_implicit_gemm import (
    sparse_submanifold_conv_bwd_input_implicit_gemm_kernel,
    sparse_submanifold_conv_bwd_weight_implicit_gemm_kernel,
)


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V', 'SPLITK', 'allow_tf32'],
)
@triton.jit
def sparse_submanifold_conv_bwd_input_implicit_gemm_splitk_kernel(
    grad_output,
    weight,
    neighbor,
    grad_input,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for N dimension
    B2: tl.constexpr,   # Block size for Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Co)
    SPLITK: tl.constexpr,  # Split K dimension
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Sparse submanifold convolution backward to input kernel using implicit GEMM.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (N, Co)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        neighbor (pointer): A pointer to the neighbor tensor of shape (N, V)
        grad_input (pointer): A pointer to the gradient of the input tensor of shape (N, Ci)
    """
    block_id_k = tl.program_id(axis=1)  # SplitK dimension
    block_id = tl.program_id(axis=0)
    block_dim_ci = tl.cdiv(Ci, B2)
    block_id_ci = block_id % block_dim_ci
    block_id_n = block_id // block_dim_ci
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Co, BK)  # Number of blocks in K dimension
    k_start = tl.cdiv(num_k * V * block_id_k, SPLITK)
    k_end = tl.cdiv(num_k * V * (block_id_k + 1), SPLITK)
    offset_n = (block_id_n * B1 + tl.arange(0, B1)) % N         # (B1,)
    offset_ci = (block_id_ci * B2 + tl.arange(0, B2)) % Ci      # (B2,)
    offset_k = tl.arange(0, BK)                                 # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)    
    
    # Iterate along V*Co dimension.
    for k in range(k_start, k_end):
        v = k // num_k
        bk = k % num_k
        # Calculate pointers to grad_output matrix.
        neighbor_offset_n = tl.load(neighbor + offset_n * V + V - 1 - v)                                    # (B1,)
        grad_output_ptr = grad_output + bk * BK + (neighbor_offset_n[:, None].to(tl.int64) * Co + offset_k[None, :])     # (B1, BK)
        # Calculate pointers to weight matrix.
        weight_ptr = weight + (((offset_k[:, None] + bk * BK) * V + v) * Ci + offset_ci[None, :])           # (BK, B2)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset_n != 0xffffffff
        k_mask = offset_k < Co - bk * BK
        grad_output_block = tl.load(grad_output_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                              # (B1, B2)
                
    # Write back the block of the output matrix with masks.
    grad_input_offset_n = block_id_n * B1 + tl.arange(0, B1)
    grad_input_offset_ci = block_id_ci * B2 + tl.arange(0, B2)
    grad_input_ptr = grad_input + block_id_k * N * Ci + (grad_input_offset_n[:, None] * Ci + grad_input_offset_ci[None, :])
    grad_input_mask = (grad_input_offset_n[:, None] < N) & (grad_input_offset_ci[None, :] < Ci)
    tl.store(grad_input_ptr, accumulator, mask=grad_input_mask)


heuristics = {
    'BV': lambda meta: max(1, meta['B2'] // meta['Ci']),
    'BCi': lambda meta: min(meta['Ci'], meta['B2']),
}

    
@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V', 'SPLITK', 'allow_tf32'],
)
@triton.heuristics(heuristics)
@triton.jit
def sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk_kernel(
    grad_output,
    input,
    neighbor,
    grad_weight,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for Co dimension
    B2: tl.constexpr,   # Block size for V * Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (N)
    BV: tl.constexpr,   # Block size for V dimension
    BCi: tl.constexpr,  # Block size for Ci dimension
    SPLITK: tl.constexpr,  # Split K dimension
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
):
    """
    Sparse submanifold convolution backward to weight kernel using implicit GEMM.
    
    Args:
        grad_output (pointer): A pointer to the gradient of the output tensor of shape (N, Co)
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        neighbor (pointer): A pointer to the neighbor tensor of shape (N, V)
        grad_weight (pointer): A pointer to the gradient of the weight tensor of shape (Co, V, Ci)
    """
    block_id_co = tl.program_id(axis=0)
    block_id_vci = tl.program_id(axis=1)
    block_id_k = tl.program_id(axis=2)
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(N, BK)  # Number of blocks in K dimension
    k_start = tl.cdiv(num_k * block_id_k, SPLITK)
    k_end = tl.cdiv(num_k * (block_id_k + 1), SPLITK)
    offset_co = (block_id_co * B1 + tl.arange(0, B1)) % Co                          # (B1,)
    offset_v = (tl.arange(0, BV) + (block_id_vci // (Ci // BCi)) * BV) % V          # (BV,)
    offset_ci = (tl.arange(0, BCi) + (block_id_vci % (Ci // BCi)) * BCi) % Ci       # (BCi,)
    offset_k = tl.arange(0, BK)                                                     # (BK,)
    neighbor_ptr = neighbor + k_start * BK * V + (offset_k[:, None] * V + offset_v[None, :])            # (BK, BV)
    grad_output_ptr = grad_output + k_start * BK * Co + (offset_k[None, :] * Co + offset_co[:, None])   # (B1, BK)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, BV * BCi), dtype=tl.float32)    
    
    # Iterate along V*Ci dimension.
    for k in range(k_start, k_end):
        mask = offset_k < N - k * BK
        # Calculate pointers to input matrix.
        input_offset_n = tl.load(neighbor_ptr, mask=mask[:, None], other=0xffffffff)            # (BK, BV)
        input_ptr = input + (input_offset_n[:, :, None].to(tl.int64) * Ci + offset_ci[None, None, :])        # (BK, BV, BCi)
        # Load the next block of input and weight.
        grad_output_block = tl.load(grad_output_ptr, mask=mask[None, :], other=0.0)
        input_block = tl.load(input_ptr, mask=input_offset_n[:, :, None] != 0xffffffff, other=0.0).reshape(BK, BV * BCi)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, input_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                  # (B1, B2)
        # Advance pointers.
        grad_output_ptr += BK * Co
        neighbor_ptr += BK * V
                
    # Write back the block of the output matrix with masks.
    grad_weight_offset_co = block_id_co * B1 + tl.arange(0, B1)
    grad_weight_offset_vci = block_id_vci * BV * BCi + tl.arange(0, BV * BCi)
    grad_weight_ptr = grad_weight + block_id_k * Co * V * Ci + (grad_weight_offset_co[:, None] * V * Ci + grad_weight_offset_vci[None, :])
    grad_weight_mask = (grad_weight_offset_co[:, None] < Co) & (grad_weight_offset_vci[None, :] < V * Ci)
    tl.store(grad_weight_ptr, accumulator, mask=grad_weight_mask)
    

def sparse_submanifold_conv_bwd_input_implicit_gemm_splitk_configs(grad_output, weight, neighbor):
    N, Ci = neighbor.shape[0], weight.shape[-1]
    MAX_NB1 = (N + 128 - 1) // 128
    MAX_NB2 = (Ci + 128 - 1) // 128
    NUM_BLOCKS = MAX_NB1 * MAX_NB2
    MIN_NUM_BLOCKS = get_num_sm()
    MAX_NUM_BLOCKS = 32 * get_num_sm()
    MIN_NUM_BLOCKS_LOG2 = max(0, int(math.log2(MIN_NUM_BLOCKS / NUM_BLOCKS)))
    MAX_NUM_BLOCKS_LOG2 = max(1, int(math.log2(MAX_NUM_BLOCKS / NUM_BLOCKS) + 1))
    configs = []
    for i in range(MIN_NUM_BLOCKS_LOG2, MAX_NUM_BLOCKS_LOG2):
        configs.append({'SPLITK': 2 ** i})
    return configs


def sparse_submanifold_conv_bwd_input_implicit_gemm_splitk_keys(grad_output, weight, neighbor):
    N, Ci, Co, V = neighbor.shape[0], weight.shape[-1], weight.shape[0], weight.shape[1]
    return f'(2^{int(math.log2(N))}, {Ci}, {Co}, {V})'


@autotune(
    config_fn=sparse_submanifold_conv_bwd_input_implicit_gemm_splitk_configs,
    key_fn=sparse_submanifold_conv_bwd_input_implicit_gemm_splitk_keys,
)
def sparse_submanifold_conv_bwd_input_implicit_gemm_splitk(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    neighbor: torch.Tensor,
    SPLITK: int = 1,
) -> torch.Tensor:
    N, Ci, Co, V = neighbor.shape[0], weight.shape[-1], weight.shape[0], weight.shape[1]
    LOGN = int(math.log2(N))
    # Launch the kernel.
    if SPLITK == 1:
        grad_input = torch.empty((N, Ci), device=weight.device, dtype=weight.dtype)
        grid = lambda META: (triton.cdiv(Ci, META['B2']) * triton.cdiv(N, META['B1']),)
        sparse_submanifold_conv_bwd_input_implicit_gemm_kernel[grid](
            grad_output,
            weight,
            neighbor,
            grad_input,
            N, LOGN, Ci, Co, V,
            allow_tf32=config.allow_tf32,
        )
        return grad_input
    else:
        grad_input = torch.empty((SPLITK, N, Ci), device=weight.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(Ci, META['B2']) * triton.cdiv(N, META['B1']), SPLITK)
        sparse_submanifold_conv_bwd_input_implicit_gemm_splitk_kernel[grid](
            grad_output,
            weight,
            neighbor,
            grad_input,
            N, LOGN, Ci, Co, V,
            SPLITK=SPLITK,
            allow_tf32=config.allow_tf32,
        )
        return grad_input.sum(0).to(weight.dtype)
    
    
def sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk_configs(grad_output, input, neighbor):
    Co, V, Ci = grad_output.shape[1], neighbor.shape[1], input.shape[1]
    MAX_NB1 = (Co + 128 - 1) // 128
    MAX_NB2 = (V * Ci + 128 - 1) // 128
    NUM_BLOCKS = MAX_NB1 * MAX_NB2
    MIN_NUM_BLOCKS = get_num_sm()
    MAX_NUM_BLOCKS = 32 * get_num_sm()
    MIN_NUM_BLOCKS_LOG2 = max(0, int(math.log2(MIN_NUM_BLOCKS / NUM_BLOCKS)))
    MAX_NUM_BLOCKS_LOG2 = max(1, int(math.log2(MAX_NUM_BLOCKS / NUM_BLOCKS) + 1))
    configs = []
    for i in range(MIN_NUM_BLOCKS_LOG2, MAX_NUM_BLOCKS_LOG2):
        configs.append({'SPLITK': 2 ** i})
    return configs


def sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk_keys(grad_output, input, neighbor):
    N, Ci, Co, V = neighbor.shape[0], input.shape[1], grad_output.shape[1], neighbor.shape[1]
    return f'(2^{int(math.log2(N))}, {Ci}, {Co}, {V})'


@autotune(
    config_fn=sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk_configs,
    key_fn=sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk_keys,
)
def sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    neighbor: torch.Tensor,
    SPLITK: int = 1,
) -> torch.Tensor:
    N, Ci, Co, V = neighbor.shape[0], input.shape[1], grad_output.shape[1], neighbor.shape[1]
    LOGN = int(math.log2(N))
    # Launch the kernel.
    if SPLITK == 1:
        grad_weight = torch.empty((Co, V, Ci), device=grad_output.device, dtype=grad_output.dtype)
        grid = lambda META: (triton.cdiv(Co, META['B1']), triton.cdiv(V * Ci, META['B2']))
        sparse_submanifold_conv_bwd_weight_implicit_gemm_kernel[grid](
            grad_output,
            input,
            neighbor,
            grad_weight,
            N, LOGN, Ci, Co, V,
            allow_tf32=config.allow_tf32,
        )
        return grad_weight
    else:
        grad_weight = torch.empty((SPLITK, Co, V, Ci), device=grad_output.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(Co, META['B1']), triton.cdiv(V * Ci, META['B2']), SPLITK)
        sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk_kernel[grid](
            grad_output,
            input,
            neighbor,
            grad_weight,
            N, LOGN, Ci, Co, V,
            SPLITK=SPLITK,
            allow_tf32=config.allow_tf32,
        )
        return grad_weight.sum(0).to(grad_output.dtype)


def sparse_submanifold_conv_bwd_implicit_gemm_splitk(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
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
        grad_input = sparse_submanifold_conv_bwd_input_implicit_gemm_splitk(
            grad_output,
            weight,
            neighbor,
        )
        
    # Grad for weight
    if weight.requires_grad:
        grad_weight = sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk(
            grad_output,
            input,
            neighbor,
        )
        
    # Grad for bias
    if bias is not None and bias.requires_grad:
        grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias
