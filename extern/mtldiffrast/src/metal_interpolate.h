// Metal interpolation
#pragma once
#import <Metal/Metal.h>
#import <torch/extension.h>

namespace mtldiffrast {

// Interpolate vertex attributes using rasterization output
// attr: [V, A] float - vertex attributes
// rast: [H, W, 4] float - rasterizer output (u, v, z/w, tri_id)
// tri: [T, 3] int - triangle indices
// Returns: (out [H, W, A], out_da [H, W, 2*A]) or (out, None)
std::tuple<torch::Tensor, torch::Tensor> interpolate(
    const torch::Tensor& attr,
    const torch::Tensor& rast,
    const torch::Tensor& tri,
    const torch::Tensor& rast_db = torch::Tensor(),
    bool enable_da = false
);

// Interpolate backward — compute attr & rast gradients.
// Returns: (grad_attr [V, A], grad_rast [H, W, 4])
std::tuple<torch::Tensor, torch::Tensor> interpolate_grad(
    const torch::Tensor& attr,
    const torch::Tensor& rast,
    const torch::Tensor& tri,
    const torch::Tensor& dy
);

// Interpolate backward with pixel derivatives.
// Returns: (grad_attr [V, A], grad_rast [H, W, 4], grad_rast_db [H, W, 4])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolate_grad_da(
    const torch::Tensor& attr,
    const torch::Tensor& rast,
    const torch::Tensor& tri,
    const torch::Tensor& dy,
    const torch::Tensor& rast_db,
    const torch::Tensor& dda
);

} // namespace mtldiffrast
