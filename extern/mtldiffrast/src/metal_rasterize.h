// Metal rasterization context and implementation
#pragma once

#import <Metal/Metal.h>
#import <torch/extension.h>

namespace mtldiffrast {

class MtlRasterizeContext {
public:
    MtlRasterizeContext();
    ~MtlRasterizeContext();

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    // Rasterize triangles in UV space
    // pos: [V, 4] float (x, y, z, w in clip space)
    // tri: [T, 3] int (vertex indices)
    // resolution: [H, W]
    // Returns: (rast [H, W, 4], rast_db [H, W, 4])
    std::tuple<torch::Tensor, torch::Tensor> rasterize(
        const torch::Tensor& pos,
        const torch::Tensor& tri,
        const std::vector<int>& resolution
    );

    // Rasterize backward — compute position gradients.
    // pos: [V, 4], tri: [T, 3], rast_out: [H, W, 4], dy: [H, W, 4], ddb: [H, W, 4]
    // Returns: grad_pos [V, 4]
    torch::Tensor rasterize_grad(
        const torch::Tensor& pos,
        const torch::Tensor& tri,
        const torch::Tensor& rast_out,
        const torch::Tensor& dy,
        const torch::Tensor& ddb
    );

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;

    // Render pipeline for hardware rasterization
    id<MTLRenderPipelineState> render_pipeline_;
    id<MTLDepthStencilState> depth_stencil_;

    // Compute pipeline fallback
    id<MTLComputePipelineState> compute_pipeline_;

    void setup_render_pipeline();
    void setup_compute_pipeline();
};

} // namespace mtldiffrast
