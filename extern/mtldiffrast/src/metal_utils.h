// Shared Metal device/queue/library + MPS zero-copy buffer binding.
// Consolidates singletons used across rasterize, interpolate, texture, antialias.
//
// MPS tensors own MTLBuffers allocated by PyTorch's MPS allocator. We wrap
// them zero-copy via at::native::mps::getMTLBufferStorage, and encode our
// kernel into PyTorch's active MPSStream so work sequences correctly with
// surrounding torch.mps ops — no per-call CPU sync, no CPU round-trip.
//
// CPU tensors wrap via newBufferWithBytesNoCopy against the tensor's data_ptr
// (Apple Silicon unified memory makes this a metadata-only operation), and
// fall back to commit + waitUntilCompleted since the output storage must be
// valid on return.
#pragma once

#import <Metal/Metal.h>
#import <torch/extension.h>
#include <string>
#include <unordered_map>

// Public MPS API for synchronization
#if __has_include(<torch/mps.h>)
#include <torch/mps.h>
#import <ATen/mps/MPSStream.h>
#define MTLDIFFRAST_HAS_MPS 1

// Forward-declared — defined in ATen/native/mps/OperationUtils.h but that
// header pulls in non-ARC MPS graph headers.
namespace at { namespace native { namespace mps {
static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}
}}}
#else
#define MTLDIFFRAST_HAS_MPS 0
#endif

#include <dlfcn.h>

namespace mtldiffrast {

//------------------------------------------------------------------------
// Shared Metal device / command queue / library singletons.

inline id<MTLDevice> mtl_get_device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

inline id<MTLCommandQueue> mtl_get_queue() {
    static id<MTLCommandQueue> q = [mtl_get_device() newCommandQueue];
    return q;
}

// Anchor symbol for dladdr-based metallib discovery.
static void _mtl_utils_anchor() {}

inline id<MTLLibrary> mtl_get_library() {
    static id<MTLLibrary> lib = nil;
    if (!lib) {
        NSError* error = nil;
        Dl_info dl_info;
        dladdr((void*)&_mtl_utils_anchor, &dl_info);
        NSString* soPath = [NSString stringWithUTF8String:dl_info.dli_fname];
        NSString* dir = [soPath stringByDeletingLastPathComponent];

        // Try combined metallib next to .so
        NSString* libPath = [dir stringByAppendingPathComponent:@"mtldiffrast.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            lib = [mtl_get_device() newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        }
        if (!lib) lib = [mtl_get_device() newDefaultLibrary];
        TORCH_CHECK(lib != nil, "[mtldiffrast] Failed to load Metal library");
    }
    return lib;
}

inline id<MTLComputePipelineState> mtl_get_pipeline(const char* name) {
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    std::string key(name);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    NSError* error = nil;
    id<MTLFunction> func = [mtl_get_library() newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    TORCH_CHECK(func != nil, "[mtldiffrast] Metal function not found: ", name);
    id<MTLComputePipelineState> pso = [mtl_get_device() newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso != nil, "[mtldiffrast] Failed to create pipeline for: ", name);
    cache[key] = pso;
    return pso;
}

//------------------------------------------------------------------------
// Tensor device helpers.

inline bool tensor_is_mps(const torch::Tensor& t) {
    return t.device().type() == torch::kMPS;
}

//------------------------------------------------------------------------
// Zero-copy Metal buffer binding.
// Apple Silicon unified memory: both MPS and CPU tensor data_ptr() are
// valid MTLBuffer-compatible addresses. No copies needed.

struct MtlBufferRef {
    id<MTLBuffer> buffer;
    NSUInteger offset;  // byte offset into buffer
};

inline MtlBufferRef tensor_to_mtl_buffer(const torch::Tensor& t) {
    TORCH_CHECK(t.is_contiguous(), "[mtldiffrast] tensor must be contiguous for Metal binding");
    TORCH_CHECK(t.numel() > 0, "[mtldiffrast] cannot create Metal buffer from empty tensor");

#if MTLDIFFRAST_HAS_MPS
    if (tensor_is_mps(t)) {
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
        TORCH_CHECK(buf != nil, "[mtldiffrast] Failed to get MPS MTLBuffer");
        NSUInteger offset = (NSUInteger)t.storage_offset() * t.element_size();
        return {buf, offset};
    }
#endif

    // CPU tensors: wrap existing memory directly (unified memory → zero copy).
    id<MTLBuffer> buf = [mtl_get_device() newBufferWithBytesNoCopy:t.data_ptr()
                                                             length:t.nbytes()
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
    TORCH_CHECK(buf != nil, "[mtldiffrast] Failed to create Metal buffer from tensor");
    return {buf, 0};
}

// Output allocation: respects the reference tensor's device so both MPS and
// CPU paths can run zero-copy end-to-end. On the broken-MPS-kernels PyTorch
// build used by shivam, `torch::zeros`/`torch::empty` direct-on-MPS may
// still raise DispatchStub-missing errors for certain dtypes; we defensively
// stage through CPU in that case and transfer — on Apple Silicon unified
// memory this is a metadata operation, so the cost vs a native MPS empty is
// negligible.
//
// The render-pipeline rasterize path still uses [MTLTexture getBytes:...]
// which requires CPU-mapped memory; the blit-encoder rewrite that removes
// that constraint is tracked in FOLLOWUPS.md. For compute paths the zero-copy
// tensor_to_mtl_buffer binding already handles both CPU and MPS.
inline torch::Tensor _safe_zeros_on_device(
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    c10::Device device
) {
    auto opts = torch::TensorOptions().dtype(dtype);
    if (device.is_cpu()) {
        return torch::zeros(sizes, opts);
    }
    try {
        return torch::zeros(sizes, opts.device(device));
    } catch (const std::exception&) {
        return torch::zeros(sizes, opts).to(device);
    }
}

inline torch::Tensor _safe_empty_on_device(
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    c10::Device device
) {
    auto opts = torch::TensorOptions().dtype(dtype);
    if (device.is_cpu()) {
        return torch::empty(sizes, opts);
    }
    try {
        return torch::empty(sizes, opts.device(device));
    } catch (const std::exception&) {
        return torch::empty(sizes, opts).to(device);
    }
}

inline torch::Tensor make_output_tensor(
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    const torch::Tensor& reference_tensor
) {
    return _safe_zeros_on_device(sizes, dtype, reference_tensor.device());
}

inline torch::Tensor make_empty_tensor(
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    const torch::Tensor& reference_tensor
) {
    return _safe_empty_on_device(sizes, dtype, reference_tensor.device());
}

// Check if any tensor in a parameter pack is on MPS.
template<typename... Ts>
inline bool any_tensor_on_mps(const Ts&... tensors) {
    bool result = false;
    (void)std::initializer_list<int>{(result = result || tensor_is_mps(tensors), 0)...};
    return result;
}

// Synchronize MPS command stream before Metal dispatch.
// Must be called when reading from MPS tensors in custom Metal kernels.
inline void mps_sync() {
#if MTLDIFFRAST_HAS_MPS
    torch::mps::commit();
#endif
}

// Dispatch helper that picks the right execution mode for the tensor's
// device. On MPS, encodes into PyTorch's active MPSStream command buffer so
// our kernel orders correctly with surrounding torch.mps ops without a CPU
// sync per call — PyTorch commits the buffer at its next sync point. On CPU,
// commits + waits synchronously since the output storage must be valid on
// return.
//
// The callback receives a ready-to-encode MTLComputeCommandEncoder; call it
// with your PSO + buffer bindings + dispatchThreads/dispatchThreadgroups.
inline void dispatch_kernel(bool on_mps,
                            void (^encode)(id<MTLCommandBuffer>,
                                           id<MTLComputeCommandEncoder>)) {
#if MTLDIFFRAST_HAS_MPS
    if (on_mps) {
        auto* stream = at::mps::getCurrentMPSStream();
        at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
                stream->endKernelCoalescing();
                id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                encode(cmdBuf, enc);
                [enc endEncoding];
            }
        });
        return;
    }
#endif
    id<MTLCommandBuffer> cmdBuf = [mtl_get_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    encode(cmdBuf, enc);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

// Batched variant for kernels that need to issue multiple dispatches within
// the same command buffer (e.g. mipmap builder). The callback receives the
// command buffer — create encoders yourself, one per dispatch, end each
// before creating the next. On MPS, one encode into the stream; on CPU, one
// commit+wait at the end.
inline void dispatch_batch(bool on_mps,
                           void (^encode_batch)(id<MTLCommandBuffer>)) {
#if MTLDIFFRAST_HAS_MPS
    if (on_mps) {
        auto* stream = at::mps::getCurrentMPSStream();
        at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
                stream->endKernelCoalescing();
                id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
                encode_batch(cmdBuf);
            }
        });
        return;
    }
#endif
    id<MTLCommandBuffer> cmdBuf = [mtl_get_queue() commandBuffer];
    encode_batch(cmdBuf);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace mtldiffrast
