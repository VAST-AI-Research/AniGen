// Metal device, command queue, and compute pipeline state management
#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <torch/extension.h>

namespace mtlmesh {

#define MTL_CHECK(call) \
do { \
    if (!(call)) { \
        TORCH_CHECK(false, "[MtlMesh] Metal error at ", __FILE__, ":", __LINE__); \
    } \
} while (0)

class MetalContext {
public:
    static MetalContext& instance();

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    // Get or create a compute pipeline state for a kernel function
    id<MTLComputePipelineState> pipeline(const std::string& kernel_name);

    // Execute a compute kernel synchronously
    void dispatch(const std::string& kernel_name,
                  const std::function<void(id<MTLComputeCommandEncoder>)>& encode,
                  uint32_t thread_count);

    // Execute with 2D grid
    void dispatch2d(const std::string& kernel_name,
                    const std::function<void(id<MTLComputeCommandEncoder>)>& encode,
                    uint32_t width, uint32_t height);

    // Wait for all submitted work to complete
    void synchronize();

private:
    MetalContext();
    ~MetalContext() = default;
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::mutex mutex_;
};

} // namespace mtlmesh
