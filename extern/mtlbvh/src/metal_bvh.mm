// Metal BVH host — builds BVH on CPU, dispatches queries on Metal GPU.
#import "metal_bvh.h"
#import "bvh_build.h"
#include <dlfcn.h>

#if __has_include(<torch/mps.h>)
#include <torch/mps.h>
#define MTLBVH_HAS_MPS 1
#else
#define MTLBVH_HAS_MPS 0
#endif

#if MTLBVH_HAS_MPS
#import <ATen/mps/MPSStream.h>
// Forward-declare this — defined in ATen/native/mps/OperationUtils.h but that
// header pulls in non-ARC MPS graph headers. The function is a one-line
// bit-cast of the storage pointer to MTLBuffer.
namespace at { namespace native { namespace mps {
static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}
}}}
#endif

static void _mtlbvh_anchor() {}

// ─── Construction ────────────────────────────────────────────────────────────

MtlBVHImpl::MtlBVHImpl(const torch::Tensor& vertices, const torch::Tensor& triangles) {
    device_ = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device_ != nil, "[mtlbvh] No Metal device found");
    queue_ = [device_ newCommandQueue];

    // Load metallib
    NSError* error = nil;
    Dl_info dl_info;
    dladdr((void*)&_mtlbvh_anchor, &dl_info);
    NSString* soPath = [NSString stringWithUTF8String:dl_info.dli_fname];
    NSString* dir = [soPath stringByDeletingLastPathComponent];
    NSString* libPath = [dir stringByAppendingPathComponent:@"mtlbvh.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
    }
    if (!library_) library_ = [device_ newDefaultLibrary];
    TORCH_CHECK(library_ != nil, "[mtlbvh] Failed to load Metal library");

    // Convert input to CPU numpy-compatible layout
    auto verts_cpu = vertices.detach().cpu().contiguous().to(torch::kFloat32);
    auto tris_cpu = triangles.detach().cpu().contiguous().to(torch::kInt32);

    int n_verts = verts_cpu.size(0);
    int n_tris = tris_cpu.size(0);
    TORCH_CHECK(n_tris > 8, "[mtlbvh] BVH needs at least 8 triangles, got ", n_tris);

    float* v_ptr = verts_cpu.data_ptr<float>();
    int* t_ptr = tris_cpu.data_ptr<int>();

    // Build triangle list
    std::vector<BvhTriangle> bvh_tris(n_tris);
    for (int i = 0; i < n_tris; ++i) {
        int i0 = t_ptr[i * 3 + 0];
        int i1 = t_ptr[i * 3 + 1];
        int i2 = t_ptr[i * 3 + 2];
        bvh_tris[i].a = {v_ptr[i0 * 3 + 0], v_ptr[i0 * 3 + 1], v_ptr[i0 * 3 + 2]};
        bvh_tris[i].b = {v_ptr[i1 * 3 + 0], v_ptr[i1 * 3 + 1], v_ptr[i1 * 3 + 2]};
        bvh_tris[i].c = {v_ptr[i2 * 3 + 0], v_ptr[i2 * 3 + 1], v_ptr[i2 * 3 + 2]};
        bvh_tris[i].id = (int64_t)i;
    }

    // Build BVH tree (CPU, reorders triangles)
    auto nodes = bvh_build(bvh_tris, 8);
    num_nodes_ = (uint32_t)nodes.size();
    num_triangles_ = (uint32_t)bvh_tris.size();

    // Upload to Metal buffers (eager copy)
    nodes_buf_ = [device_ newBufferWithBytes:nodes.data()
                                      length:nodes.size() * sizeof(BvhNode)
                                     options:MTLResourceStorageModeShared];
    tris_buf_ = [device_ newBufferWithBytes:bvh_tris.data()
                                     length:bvh_tris.size() * sizeof(BvhTriangle)
                                    options:MTLResourceStorageModeShared];

    // Pre-create all pipeline state objects — 4 kernels × 3 output dtypes.
    auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
        NSError* error = nil;
        id<MTLFunction> func = [library_ newFunctionWithName:
            [NSString stringWithUTF8String:name]];
        TORCH_CHECK(func != nil, "[mtlbvh] Metal function not found: ", name);
        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:func error:&error];
        TORCH_CHECK(pso != nil, "[mtlbvh] Failed to create pipeline for: ", name);
        return pso;
    };
    // Dtype suffix lookup: fp32 → "" (backward-compat), fp16 → "_half", bf16 → "_bfloat".
    // Arrays hold strong references (ARC) — unroll the outer loop so each
    // assignment targets a named ivar rather than a pointer-to-pointer (which
    // ARC rejects without explicit ownership qualifiers).
    const char* suffixes[3] = {"", "_half", "_bfloat"};
    for (int d = 0; d < 3; ++d) {
        pso_unsigned_[d] = make_pso(
            (std::string("unsigned_distance_kernel") + suffixes[d]).c_str());
        pso_signed_watertight_[d] = make_pso(
            (std::string("signed_distance_watertight_kernel") + suffixes[d]).c_str());
        pso_signed_raystab_[d] = make_pso(
            (std::string("signed_distance_raystab_kernel") + suffixes[d]).c_str());
        pso_raytrace_[d] = make_pso(
            (std::string("raytrace_kernel") + suffixes[d]).c_str());
    }
}

// Resolve an OutDtype enum to a torch ScalarType for output-tensor allocation.
static torch::ScalarType out_dtype_to_scalar(int out_dtype) {
    switch (out_dtype) {
        case MtlBVHImpl::kFp32: return torch::kFloat32;
        case MtlBVHImpl::kFp16: return torch::kFloat16;
        case MtlBVHImpl::kBf16: return torch::kBFloat16;
    }
    TORCH_CHECK(false, "[mtlbvh] Invalid out_dtype (expected 0/1/2): ", out_dtype);
}

// ─── Helper: create Metal buffer from tensor (zero-copy on Apple Silicon) ────

// Returns (buffer, offset). `offset` accounts for non-zero storage_offset on
// MPS tensors, which view operations produce. CPU tensors always have offset 0
// because we create fresh buffers from contiguous storage.
struct TensorBuf { id<MTLBuffer> buffer; NSUInteger offset; };

static TensorBuf tensor_to_buffer(id<MTLDevice> dev, const torch::Tensor& t) {
    TORCH_CHECK(t.numel() > 0, "[mtlbvh] empty tensor");
#if MTLBVH_HAS_MPS
    if (t.device().is_mps()) {
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
        TORCH_CHECK(buf != nil, "[mtlbvh] Failed to get MPS MTLBuffer");
        NSUInteger offset = (NSUInteger)t.storage_offset() * t.element_size();
        return {buf, offset};
    }
#endif
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:t.data_ptr()
                                               length:t.nbytes()
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    TORCH_CHECK(buf != nil, "[mtlbvh] Failed to create buffer");
    return {buf, 0};
}

// Dispatch a pre-built command setup. On MPS, encodes into PyTorch's active
// MPSStream and returns without waiting (PyTorch commits at its next sync
// point). On CPU, commits and waits synchronously since the output tensor's
// storage must be valid on return.
static void dispatch_query(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                           id<MTLComputePipelineState> pso,
                           bool on_mps,
                           void (^encode)(id<MTLComputeCommandEncoder>)) {
#if MTLBVH_HAS_MPS
    if (on_mps) {
        auto* stream = at::mps::getCurrentMPSStream();
        at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pso];
                encode(enc);
                [enc endEncoding];
            }
        });
        return;
    }
#endif
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode(enc);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

// ─── Unsigned distance ──────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MtlBVHImpl::unsigned_distance(const torch::Tensor& positions, bool return_uvw, int out_dtype) {
    bool on_mps = positions.device().is_mps();
    auto dev_opts = torch::TensorOptions().device(positions.device());

    auto pos_f = positions.detach().contiguous().to(torch::kFloat32);
    auto prefix = pos_f.sizes().vec();
    prefix.pop_back();  // remove last dim (3)
    int64_t N = pos_f.numel() / 3;

    pos_f = pos_f.view({N, 3});

    auto out_scalar = out_dtype_to_scalar(out_dtype);
    auto distances = torch::empty({N}, dev_opts.dtype(out_scalar));
    auto face_id = torch::empty({N}, dev_opts.dtype(torch::kInt64));
    // zeros({1,3}) is unused when !return_uvw but buffer index 3 still needs
    // a valid binding. Build on CPU and move — some PyTorch MPS builds have a
    // broken torch::zeros fp32 MPS kernel (DispatchStub missing).
    torch::Tensor uvw;
    if (return_uvw) {
        uvw = torch::empty({N, 3}, dev_opts.dtype(torch::kFloat32));
    } else if (on_mps) {
        uvw = torch::zeros({1, 3}, torch::kFloat32).to(positions.device());
    } else {
        uvw = torch::zeros({1, 3}, torch::kFloat32);
    }

    uint32_t n_elements = (uint32_t)N;
    uint32_t ret_uvw = return_uvw ? 1 : 0;

    auto pos_buf = tensor_to_buffer(device_, pos_f);
    auto dist_buf = tensor_to_buffer(device_, distances);
    auto fid_buf = tensor_to_buffer(device_, face_id);
    auto uvw_buf = tensor_to_buffer(device_, uvw);

    auto pso = pso_unsigned_[out_dtype];
    NSUInteger tw = pso.threadExecutionWidth;
    dispatch_query(device_, queue_, pso, on_mps, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:pos_buf.buffer  offset:pos_buf.offset  atIndex:0];
        [enc setBuffer:dist_buf.buffer offset:dist_buf.offset atIndex:1];
        [enc setBuffer:fid_buf.buffer  offset:fid_buf.offset  atIndex:2];
        [enc setBuffer:uvw_buf.buffer  offset:uvw_buf.offset  atIndex:3];
        [enc setBuffer:nodes_buf_ offset:0 atIndex:4];
        [enc setBuffer:tris_buf_  offset:0 atIndex:5];
        [enc setBytes:&n_elements length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&ret_uvw    length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    });

    // Reshape outputs back to original prefix shape
    distances = distances.view(prefix);
    face_id = face_id.view(prefix);
    if (return_uvw) {
        auto uvw_shape = prefix;
        uvw_shape.push_back(3);
        uvw = uvw.view(uvw_shape);
    } else {
        uvw = torch::Tensor();  // None
    }

    return {distances, face_id, uvw};
}

// ─── Signed distance ────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MtlBVHImpl::signed_distance(const torch::Tensor& positions, bool return_uvw, int mode, int out_dtype) {
    bool on_mps = positions.device().is_mps();
    auto dev_opts = torch::TensorOptions().device(positions.device());

    auto pos_f = positions.detach().contiguous().to(torch::kFloat32);
    auto prefix = pos_f.sizes().vec();
    prefix.pop_back();
    int64_t N = pos_f.numel() / 3;

    pos_f = pos_f.view({N, 3});

    auto out_scalar = out_dtype_to_scalar(out_dtype);
    auto distances = torch::empty({N}, dev_opts.dtype(out_scalar));
    auto face_id = torch::empty({N}, dev_opts.dtype(torch::kInt64));
    // zeros({1,3}) is unused when !return_uvw but buffer index 3 still needs
    // a valid binding. Build on CPU and move — some PyTorch MPS builds have a
    // broken torch::zeros fp32 MPS kernel (DispatchStub missing).
    torch::Tensor uvw;
    if (return_uvw) {
        uvw = torch::empty({N, 3}, dev_opts.dtype(torch::kFloat32));
    } else if (on_mps) {
        uvw = torch::zeros({1, 3}, torch::kFloat32).to(positions.device());
    } else {
        uvw = torch::zeros({1, 3}, torch::kFloat32);
    }

    uint32_t n_elements = (uint32_t)N;
    uint32_t ret_uvw = return_uvw ? 1 : 0;

    auto pos_buf = tensor_to_buffer(device_, pos_f);
    auto dist_buf = tensor_to_buffer(device_, distances);
    auto fid_buf = tensor_to_buffer(device_, face_id);
    auto uvw_buf = tensor_to_buffer(device_, uvw);

    auto pso = (mode == 0) ? pso_signed_watertight_[out_dtype]
                            : pso_signed_raystab_[out_dtype];
    NSUInteger tw = pso.threadExecutionWidth;
    dispatch_query(device_, queue_, pso, on_mps, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:pos_buf.buffer  offset:pos_buf.offset  atIndex:0];
        [enc setBuffer:dist_buf.buffer offset:dist_buf.offset atIndex:1];
        [enc setBuffer:fid_buf.buffer  offset:fid_buf.offset  atIndex:2];
        [enc setBuffer:uvw_buf.buffer  offset:uvw_buf.offset  atIndex:3];
        [enc setBuffer:nodes_buf_ offset:0 atIndex:4];
        [enc setBuffer:tris_buf_  offset:0 atIndex:5];
        [enc setBytes:&n_elements length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&ret_uvw    length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    });

    distances = distances.view(prefix);
    face_id = face_id.view(prefix);
    if (return_uvw) {
        auto uvw_shape = prefix;
        uvw_shape.push_back(3);
        uvw = uvw.view(uvw_shape);
    } else {
        uvw = torch::Tensor();
    }

    return {distances, face_id, uvw};
}

// ─── Ray trace ──────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MtlBVHImpl::ray_trace(const torch::Tensor& rays_o, const torch::Tensor& rays_d, int out_dtype) {
    bool on_mps = rays_o.device().is_mps();
    auto dev_opts = torch::TensorOptions().device(rays_o.device());

    auto ro_f = rays_o.detach().contiguous().to(torch::kFloat32);
    auto rd_f = rays_d.detach().contiguous().to(torch::kFloat32);

    auto prefix = ro_f.sizes().vec();
    prefix.pop_back();
    int64_t N = ro_f.numel() / 3;

    ro_f = ro_f.view({N, 3});
    rd_f = rd_f.view({N, 3});

    auto out_scalar = out_dtype_to_scalar(out_dtype);
    auto positions = torch::empty({N, 3}, dev_opts.dtype(torch::kFloat32));
    auto face_id = torch::empty({N}, dev_opts.dtype(torch::kInt64));
    auto depth = torch::empty({N}, dev_opts.dtype(out_scalar));

    uint32_t n_elements = (uint32_t)N;

    auto ro_buf = tensor_to_buffer(device_, ro_f);
    auto rd_buf = tensor_to_buffer(device_, rd_f);
    auto pos_buf = tensor_to_buffer(device_, positions);
    auto fid_buf = tensor_to_buffer(device_, face_id);
    auto dep_buf = tensor_to_buffer(device_, depth);

    auto pso = pso_raytrace_[out_dtype];
    NSUInteger tw = pso.threadExecutionWidth;
    dispatch_query(device_, queue_, pso, on_mps, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ro_buf.buffer  offset:ro_buf.offset  atIndex:0];
        [enc setBuffer:rd_buf.buffer  offset:rd_buf.offset  atIndex:1];
        [enc setBuffer:pos_buf.buffer offset:pos_buf.offset atIndex:2];
        [enc setBuffer:fid_buf.buffer offset:fid_buf.offset atIndex:3];
        [enc setBuffer:dep_buf.buffer offset:dep_buf.offset atIndex:4];
        [enc setBuffer:nodes_buf_ offset:0 atIndex:5];
        [enc setBuffer:tris_buf_  offset:0 atIndex:6];
        [enc setBytes:&n_elements length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    });

    auto pos_shape = prefix;
    pos_shape.push_back(3);
    positions = positions.view(pos_shape);
    face_id = face_id.view(prefix);
    depth = depth.view(prefix);

    return {positions, face_id, depth};
}
