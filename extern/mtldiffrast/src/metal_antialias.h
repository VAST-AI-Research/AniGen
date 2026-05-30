// Metal antialias — header with function prototypes
#pragma once
#import <Metal/Metal.h>
#import <torch/extension.h>

namespace mtldiffrast {

// Build the edge-vertex topology hash table from a triangle mesh.
// tri: [T, 3] int — triangle vertex indices
// Returns: topology hash tensor (flat int buffer)
torch::Tensor antialias_construct_topology_hash(const torch::Tensor& tri);

// Forward antialias pass.
// color: [N, H, W, C] float — input color buffer
// rast: [N, H, W, 4] float — rasterizer output (u, v, z/w, tri_id)
// pos: [V, 4] or [N, V, 4] float — vertex positions in clip space
// tri: [T, 3] int — triangle indices
// topology_hash: flat int tensor from antialias_construct_topology_hash
// Returns: (output [N, H, W, C], work_buffer)
std::tuple<torch::Tensor, torch::Tensor> antialias_fwd(
    const torch::Tensor& color,
    const torch::Tensor& rast,
    const torch::Tensor& pos,
    const torch::Tensor& tri,
    const torch::Tensor& topology_hash
);

// Backward antialias pass.
// color: [N, H, W, C] float — original input color
// rast: [N, H, W, 4] float — rasterizer output
// pos: [V, 4] or [N, V, 4] float — vertex positions
// tri: [T, 3] int — triangle indices
// dy: [N, H, W, C] float — incoming gradients
// work_buffer: from antialias_fwd
// Returns: (grad_color [N, H, W, C], grad_pos [V, 4] or [N, V, 4])
std::tuple<torch::Tensor, torch::Tensor> antialias_grad(
    const torch::Tensor& color,
    const torch::Tensor& rast,
    const torch::Tensor& pos,
    const torch::Tensor& tri,
    const torch::Tensor& dy,
    const torch::Tensor& work_buffer
);

} // namespace mtldiffrast
