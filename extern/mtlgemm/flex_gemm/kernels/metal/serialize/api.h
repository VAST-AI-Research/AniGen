#pragma once
#include <torch/extension.h>

namespace flex_gemm {
namespace metal {
namespace serialize {

void z_order_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
);

torch::Tensor z_order_decode(
    const torch::Tensor& codes,
    const size_t bit_length
);

void hilbert_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
);

torch::Tensor hilbert_decode(
    const torch::Tensor& codes,
    const size_t bit_length
);

} // namespace serialize
} // namespace metal
} // namespace flex_gemm
