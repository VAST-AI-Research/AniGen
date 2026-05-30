// Metal BVH host — builds BVH on CPU, dispatches queries on Metal GPU.
#pragma once

#import <Metal/Metal.h>
#import <torch/extension.h>
#include "bvh_types.h"
#include <vector>

class MtlBVHImpl {
public:
    MtlBVHImpl(const torch::Tensor& vertices, const torch::Tensor& triangles);

    // Output-dtype selector for scalar outputs (distances / depth).
    //   0 = float32 (default, backward-compat)
    //   1 = float16
    //   2 = bfloat16
    // Vec3 inputs/outputs (positions, uvw, rays) always stay fp32.
    enum OutDtype : int { kFp32 = 0, kFp16 = 1, kBf16 = 2 };

    // Unsigned distance: closest point on mesh.
    // Returns: (distances [N] in out_dtype, face_id [N], optional uvw [N,3] fp32)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    unsigned_distance(const torch::Tensor& positions, bool return_uvw, int out_dtype);

    // Signed distance: watertight or raystab mode.
    // Returns: (distances [N] in out_dtype, face_id [N], optional uvw [N,3] fp32)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    signed_distance(const torch::Tensor& positions, bool return_uvw, int mode, int out_dtype);

    // Ray trace: ray-triangle intersection.
    // Returns: (positions [N,3] fp32, face_id [N], depth [N] in out_dtype)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    ray_trace(const torch::Tensor& rays_o, const torch::Tensor& rays_d, int out_dtype);

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;

    // GPU buffers for BVH data (uploaded once at construction)
    id<MTLBuffer> nodes_buf_;
    id<MTLBuffer> tris_buf_;
    uint32_t num_nodes_;
    uint32_t num_triangles_;

    // Cached pipeline state objects — 4 kernels × 3 output dtypes = 12 PSOs.
    // Indexed by [out_dtype] where out_dtype ∈ {kFp32=0, kFp16=1, kBf16=2}.
    id<MTLComputePipelineState> pso_unsigned_[3];
    id<MTLComputePipelineState> pso_signed_watertight_[3];
    id<MTLComputePipelineState> pso_signed_raystab_[3];
    id<MTLComputePipelineState> pso_raytrace_[3];
};
