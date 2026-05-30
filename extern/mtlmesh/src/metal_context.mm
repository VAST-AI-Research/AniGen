// Metal context implementation
#import "metal_context.h"
#include <dlfcn.h>

namespace mtlmesh {

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device_ != nil, "[MtlMesh] No Metal device found");
    queue_ = [device_ newCommandQueue];
    TORCH_CHECK(queue_ != nil, "[MtlMesh] Failed to create command queue");

    // Load the default Metal library (compiled shaders)
    NSError* error = nil;

    // Try loading from the bundle path first (for pip install -e .)
    // The .metallib will be compiled at build time
    NSString* libPath = nil;

    // Search for metallib in common locations
    NSBundle* bundle = [NSBundle mainBundle];
    libPath = [bundle pathForResource:@"mtlmesh" ofType:@"metallib"];

    if (!libPath) {
        // Find metallib alongside the compiled .so using dladdr
        Dl_info dl_info;
        if (dladdr((void*)&MetalContext::instance, &dl_info)) {
            NSString* soPath = [NSString stringWithUTF8String:dl_info.dli_fname];
            NSString* dir = [soPath stringByDeletingLastPathComponent];
            libPath = [dir stringByAppendingPathComponent:@"cumesh.metallib"];
        }
    }

    if (libPath && [[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        library_ = [device_ newLibraryWithURL:libURL error:&error];
    }

    if (!library_) {
        // Fallback: try default library
        library_ = [device_ newDefaultLibrary];
    }

    TORCH_CHECK(library_ != nil, "[MtlMesh] Failed to load Metal library: ",
                error ? [[error localizedDescription] UTF8String] : "unknown error");
}

id<MTLComputePipelineState> MetalContext::pipeline(const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pipelines_.find(kernel_name);
    if (it != pipelines_.end()) {
        return it->second;
    }

    NSString* name = [NSString stringWithUTF8String:kernel_name.c_str()];
    id<MTLFunction> func = [library_ newFunctionWithName:name];
    TORCH_CHECK(func != nil, "[MtlMesh] Kernel '", kernel_name, "' not found in Metal library");

    NSError* error = nil;
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso != nil, "[MtlMesh] Failed to create pipeline for '", kernel_name, "': ",
                error ? [[error localizedDescription] UTF8String] : "unknown");

    pipelines_[kernel_name] = pso;
    return pso;
}

void MetalContext::dispatch(const std::string& kernel_name,
                            const std::function<void(id<MTLComputeCommandEncoder>)>& encode,
                            uint32_t thread_count) {
    if (thread_count == 0) return;

    auto pso = pipeline(kernel_name);
    NSUInteger maxThreads = [pso maxTotalThreadsPerThreadgroup];
    NSUInteger threadGroupSize = std::min((NSUInteger)256, maxThreads);

    id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso];

    encode(encoder);

    MTLSize gridSize = MTLSizeMake(thread_count, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void MetalContext::dispatch2d(const std::string& kernel_name,
                              const std::function<void(id<MTLComputeCommandEncoder>)>& encode,
                              uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;

    auto pso = pipeline(kernel_name);

    id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso];

    encode(encoder);

    MTLSize gridSize = MTLSizeMake(width, height, 1);
    MTLSize groupSize = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void MetalContext::synchronize() {
    id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace mtlmesh
