#pragma once
#include <torch/extension.h>

namespace flex_gemm {
namespace metal {
namespace grid_sample {

torch::Tensor hashmap_build_grid_sample_3d_nearest_neighbor_map(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    int W, int H, int D
);

std::tuple<torch::Tensor, torch::Tensor> hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    int W, int H, int D
);

torch::Tensor indice_weighted_sum_fwd(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& weight
);

torch::Tensor indice_weighted_sum_bwd_input(
    const torch::Tensor& grad_output,
    const torch::Tensor& indices,
    const torch::Tensor& weight,
    int64_t N
);

} // namespace grid_sample
} // namespace metal
} // namespace flex_gemm
