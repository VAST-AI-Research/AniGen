// Metal texture context — forward-pass texture sampling and mip construction.
#pragma once
#import <Metal/Metal.h>
#import <torch/extension.h>
#include <vector>

namespace mtldiffrast {

//------------------------------------------------------------------------
// Mip wrapper for texture mipmap state.

struct TextureMipWrapper
{
    torch::Tensor mip;                  // Flat mip data.
    int max_mip_level = 0;             // Max mip level.
    std::vector<int64_t> texture_size; // Original texture shape.
    bool cube_mode = false;
};

//------------------------------------------------------------------------
// Constants — must match texture.metal definitions.

#define TEX_MAX_MIP_LEVEL           16
#define TEX_MODE_NEAREST            0
#define TEX_MODE_LINEAR             1
#define TEX_MODE_LINEAR_MIPMAP_NEAREST 2
#define TEX_MODE_LINEAR_MIPMAP_LINEAR  3
#define TEX_MODE_COUNT              4
#define TEX_BOUNDARY_MODE_CUBE      0
#define TEX_BOUNDARY_MODE_WRAP      1
#define TEX_BOUNDARY_MODE_CLAMP     2
#define TEX_BOUNDARY_MODE_ZERO      3
#define TEX_BOUNDARY_MODE_COUNT     4

//------------------------------------------------------------------------
// Function prototypes.

// Build mipmap chain from base texture. Returns wrapper with mip data.
TextureMipWrapper texture_construct_mip(
    const torch::Tensor& tex,
    int max_mip_level,
    bool cube_mode
);

// Forward texture lookup (with mip support).
torch::Tensor texture_fwd_mip(
    const torch::Tensor& tex,
    const torch::Tensor& uv,
    const torch::Tensor& uv_da,
    const torch::Tensor& mip_level_bias,
    TextureMipWrapper& mip_wrapper,
    const std::vector<torch::Tensor>& mip_stack,
    int filter_mode,
    int boundary_mode
);

// Forward texture lookup (no mips).
torch::Tensor texture_fwd(
    const torch::Tensor& tex,
    const torch::Tensor& uv,
    int filter_mode,
    int boundary_mode
);

// Backward texture lookup — all filter modes.
// Returns: (grad_tex, grad_uv, grad_uv_da, grad_mip_level_bias)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> texture_grad(
    const torch::Tensor& tex,
    const torch::Tensor& uv,
    const torch::Tensor& dy,
    const torch::Tensor& uv_da,
    const torch::Tensor& mip_level_bias,
    TextureMipWrapper& mip_wrapper,
    const std::vector<torch::Tensor>& mip_stack,
    int filter_mode,
    int boundary_mode
);

} // namespace mtldiffrast
