#pragma once
#include <torch/extension.h>

namespace flex_gemm {
namespace metal {
namespace hash {

void hashmap_insert_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values
);

torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys
);

void hashmap_insert_3d_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    const torch::Tensor& values,
    int W, int H, int D
);

torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W, int H, int D
);

void hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W, int H, int D
);

} // namespace hash
} // namespace metal
} // namespace flex_gemm
