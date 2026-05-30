#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string>
#include <unordered_map>
#include <functional>

namespace flex_gemm {
namespace metal {

class MetalContext {
public:
    static MetalContext& instance();

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    id<MTLComputePipelineState> pipeline(const std::string& kernel_name);

    // Synchronous 1D dispatch
    void dispatch(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        uint64_t thread_count
    );

    // Synchronous 2D dispatch with explicit grid/threadgroup
    void dispatch_2d(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        MTLSize grid_size,
        MTLSize threadgroup_size
    );

    // Threadgroup-level dispatch (dispatchThreadgroups, not dispatchThreads).
    // For tiled GEMM kernels that use specific block sizes. CPU/private-queue path.
    // wait=false skips waitUntilCompleted for non-final dispatches in a sequence.
    void dispatch_threadgroups(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        MTLSize grid_size,
        MTLSize threadgroup_size,
        bool wait
    );

    // MPS-mode dispatch: encodes into the active PyTorch MPSStream so the
    // kernel orders correctly with surrounding ops and PyTorch handles commit.
    void dispatch_mps(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        uint64_t thread_count
    );

    void dispatch_2d_mps(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        MTLSize grid_size,
        MTLSize threadgroup_size
    );

    void dispatch_threadgroups_mps(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        MTLSize grid_size,
        MTLSize threadgroup_size
    );

    void synchronize();

    // Batched dispatch: encode multiple kernels into one command buffer
    void begin_batch();
    void end_batch();  // commits and waits

    // Dispatch that uses active batch if available, otherwise standalone
    void dispatch_batched(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        uint64_t thread_count
    );

    void dispatch_2d_batched(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        MTLSize grid_size,
        MTLSize threadgroup_size
    );

    bool in_batch() const { return batch_active_; }

private:
    MetalContext();
    ~MetalContext() = default;
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache_;

    bool batch_active_ = false;
    id<MTLCommandBuffer> batch_cmdbuf_ = nil;
};

} // namespace metal
} // namespace flex_gemm
