#pragma once
#include <torch/extension.h>

namespace flex_gemm {
namespace metal {
namespace spconv {

torch::Tensor hashmap_build_submanifold_conv_neighbour_map_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Dw, int Dh, int Dd
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_map_post_process_for_masked_implicit_gemm_1(
    const torch::Tensor& neighbor_map
);

std::tuple<torch::Tensor, torch::Tensor>
neighbor_map_post_process_for_masked_implicit_gemm_2(
    const torch::Tensor& gray_code,
    const torch::Tensor& sorted_idx,
    int block_size
);

// Implicit GEMM forward: input[N,Ci] x weight[Co,V,Ci] -> output[N,Co]
torch::Tensor spconv_fwd_implicit_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& neighbor
);

// Implicit GEMM backward: returns (grad_input[N,Ci], grad_weight[Co,V,Ci])
std::tuple<torch::Tensor, torch::Tensor> spconv_bwd_implicit_gemm(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& neighbor
);

// Masked implicit GEMM forward — uses precomputed sorted_idx + valid_kernel
// + valid_kernel_seg to iterate only the valid V positions per n-block.
torch::Tensor spconv_fwd_masked_implicit_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& neighbor,
    const torch::Tensor& sorted_idx_i64,
    const torch::Tensor& valid_kernel,
    const torch::Tensor& valid_kernel_seg
);

// Masked implicit GEMM backward: returns (grad_input[N,Ci], grad_weight[Co,V,Ci]).
// Uses the same precomputed sorted_idx / valid_kernel / valid_kernel_seg as the
// forward (identical V-set under change of variable u = V-1-v) for grad_input,
// and valid_signal_i / valid_signal_o / valid_signal_seg from
// neighbor_map_post_process_for_masked_implicit_gemm_1 for grad_weight.
std::tuple<torch::Tensor, torch::Tensor> spconv_bwd_masked_implicit_gemm(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& neighbor,
    const torch::Tensor& sorted_idx_i64,
    const torch::Tensor& valid_kernel,
    const torch::Tensor& valid_kernel_seg,
    const torch::Tensor& valid_signal_i,
    const torch::Tensor& valid_signal_o,
    const torch::Tensor& valid_signal_seg
);

} // namespace spconv
} // namespace metal
} // namespace flex_gemm
