// Metal antialias host-side implementation — zero-copy MPS path.
#import "metal_antialias.h"
#import "metal_utils.h"

namespace mtldiffrast {

//------------------------------------------------------------------------
// Constants matching the Metal shader.

#define AA_MESH_KERNEL_THREADS_PER_BLOCK            256
#define AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH         32
#define AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT        8
#define AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK        256
#define AA_GRAD_KERNEL_THREADS_PER_BLOCK            256

static inline int aa_hash_elements_per_triangle(int alloc) { return (alloc >= (2 << 25)) ? 4 : 8; }

//------------------------------------------------------------------------
// Constant-buffer params — must match AntialiasConstParams in antialias.metal.

struct AntialiasConstParams
{
    int     allocTriangles;
    int     numTriangles;
    int     numVertices;
    int     width;
    int     height;
    int     n;
    int     channels;
    float   xh, yh;
    int     instance_mode;
    int     tri_const;
};

//------------------------------------------------------------------------
// Build topology hash.

torch::Tensor antialias_construct_topology_hash(const torch::Tensor& tri)
{
    TORCH_CHECK(tri.dim() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    auto tri_c = tri.contiguous().to(torch::kInt32);

    AntialiasConstParams p = {};
    p.numTriangles = (int)tri_c.size(0);
    p.numVertices = 0x7fffffff;

    p.allocTriangles = 64;
    while (p.allocTriangles < p.numTriangles)
        p.allocTriangles <<= 1;

    int hashSize = p.allocTriangles * aa_hash_elements_per_triangle(p.allocTriangles) * 4;
    auto ev_hash = torch::zeros({hashSize}, torch::TensorOptions().dtype(torch::kInt32));

    if (any_tensor_on_mps(tri)) mps_sync();

    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto hash_ref = tensor_to_mtl_buffer(ev_hash);

    auto pso = mtl_get_pipeline("AntialiasFwdMeshKernel");
    auto queue = mtl_get_queue();
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:tri_ref.buffer  offset:tri_ref.offset  atIndex:0];
    [enc setBuffer:hash_ref.buffer offset:hash_ref.offset atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];

    NSUInteger threadCount = (NSUInteger)p.numTriangles;
    NSUInteger threadGroupSize = MIN((NSUInteger)AA_MESH_KERNEL_THREADS_PER_BLOCK, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(threadCount, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return ev_hash;
}

//------------------------------------------------------------------------
// Forward pass.

std::tuple<torch::Tensor, torch::Tensor> antialias_fwd(
    const torch::Tensor& color,
    const torch::Tensor& rast,
    const torch::Tensor& pos,
    const torch::Tensor& tri,
    const torch::Tensor& topology_hash
)
{
    TORCH_CHECK(color.dim() == 4, "color must have shape [N, H, W, C]");
    TORCH_CHECK(rast.dim() == 4 && rast.size(3) == 4, "rast must have shape [N, H, W, 4]");
    TORCH_CHECK(tri.dim() == 2 && tri.size(1) == 3, "tri must have shape [T, 3]");

    int instance_mode = (pos.dim() > 2) ? 1 : 0;

    auto color_c = color.contiguous().to(torch::kFloat32);
    auto rast_c = rast.contiguous().to(torch::kFloat32);
    auto pos_c = pos.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32);
    auto hash_c = topology_hash.contiguous().to(torch::kInt32);

    AntialiasConstParams p = {};
    p.instance_mode = instance_mode;
    p.numVertices  = (int)pos_c.size(instance_mode ? 1 : 0);
    p.numTriangles = (int)tri_c.size(0);
    p.n            = (int)color_c.size(0);
    p.height       = (int)color_c.size(1);
    p.width        = (int)color_c.size(2);
    p.channels     = (int)color_c.size(3);
    p.xh = .5f * (float)p.width;
    p.yh = .5f * (float)p.height;

    p.allocTriangles = 64;
    while (p.allocTriangles < p.numTriangles)
        p.allocTriangles <<= 1;

    auto out = color_c.clone();

    int workBufElems = p.n * p.width * p.height * 8 + 4;
    auto work_buffer = torch::zeros({workBufElems}, torch::TensorOptions().dtype(torch::kInt32));

    auto queue = mtl_get_queue();

    if (any_tensor_on_mps(color, rast, pos, tri)) mps_sync();

    auto color_ref = tensor_to_mtl_buffer(color_c);
    auto rast_ref = tensor_to_mtl_buffer(rast_c);
    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto pos_ref = tensor_to_mtl_buffer(pos_c);
    auto out_ref = tensor_to_mtl_buffer(out);
    auto work_ref = tensor_to_mtl_buffer(work_buffer);
    auto hash_ref = tensor_to_mtl_buffer(hash_c);

    // --- Kernel 1: Discontinuity finder ---
    {
        auto pso = mtl_get_pipeline("AntialiasFwdDiscontinuityKernel");
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:rast_ref.buffer offset:rast_ref.offset atIndex:0];
        [enc setBuffer:work_ref.buffer offset:work_ref.offset atIndex:1];
        [enc setBuffer:work_ref.buffer offset:work_ref.offset atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];

        MTLSize grid = MTLSizeMake((NSUInteger)p.width, (NSUInteger)p.height, (NSUInteger)p.n);
        NSUInteger tgW = MIN(32u, pso.maxTotalThreadsPerThreadgroup);
        NSUInteger tgH = MIN(8u, pso.maxTotalThreadsPerThreadgroup / tgW);
        MTLSize tg = MTLSizeMake(tgW, tgH, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    // --- Kernel 2: Analysis ---
    {
        int workCount = work_buffer.data_ptr<int>()[0];
        if (workCount > 0)
        {
            bool on_mps = out.device().is_mps();
            auto pso = mtl_get_pipeline("AntialiasFwdAnalysisKernel");
            dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
                (void)cmdBuf;
                [enc setComputePipelineState:pso];
                [enc setBuffer:color_ref.buffer offset:color_ref.offset atIndex:0];
                [enc setBuffer:rast_ref.buffer  offset:rast_ref.offset  atIndex:1];
                [enc setBuffer:tri_ref.buffer   offset:tri_ref.offset   atIndex:2];
                [enc setBuffer:pos_ref.buffer   offset:pos_ref.offset   atIndex:3];
                [enc setBuffer:out_ref.buffer   offset:out_ref.offset   atIndex:4];
                [enc setBuffer:work_ref.buffer  offset:work_ref.offset  atIndex:5];
                [enc setBuffer:hash_ref.buffer  offset:hash_ref.offset  atIndex:6];
                [enc setBytes:&p length:sizeof(p) atIndex:7];

                NSUInteger threadGroupSize = MIN((NSUInteger)AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK,
                                                 pso.maxTotalThreadsPerThreadgroup);
                [enc dispatchThreads:MTLSizeMake((NSUInteger)workCount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
            });
        }
    }

    return {out, work_buffer};
}

//------------------------------------------------------------------------
// Backward pass.

std::tuple<torch::Tensor, torch::Tensor> antialias_grad(
    const torch::Tensor& color,
    const torch::Tensor& rast,
    const torch::Tensor& pos,
    const torch::Tensor& tri,
    const torch::Tensor& dy,
    const torch::Tensor& work_buffer
)
{
    TORCH_CHECK(color.dim() == 4, "color must have shape [N, H, W, C]");
    TORCH_CHECK(rast.dim() == 4 && rast.size(3) == 4, "rast must have shape [N, H, W, 4]");
    TORCH_CHECK(tri.dim() == 2 && tri.size(1) == 3, "tri must have shape [T, 3]");
    TORCH_CHECK(dy.dim() == 4, "dy must have shape [N, H, W, C]");

    int instance_mode = (pos.dim() > 2) ? 1 : 0;

    auto color_c = color.contiguous().to(torch::kFloat32);
    auto rast_c = rast.contiguous().to(torch::kFloat32);
    auto pos_c = pos.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32);
    auto dy_c = dy.contiguous().to(torch::kFloat32);
    auto work_c = work_buffer.contiguous().to(torch::kInt32);

    AntialiasConstParams p = {};
    p.instance_mode = instance_mode;
    p.numVertices  = (int)pos_c.size(instance_mode ? 1 : 0);
    p.numTriangles = (int)tri_c.size(0);
    p.n            = (int)color_c.size(0);
    p.height       = (int)color_c.size(1);
    p.width        = (int)color_c.size(2);
    p.channels     = (int)color_c.size(3);
    p.xh = .5f * (float)p.width;
    p.yh = .5f * (float)p.height;

    auto grad_color = dy_c.clone();
    auto grad_pos = torch::zeros_like(pos_c);

    int workCount = work_c.data_ptr<int>()[0];
    if (workCount <= 0)
        return {grad_color, grad_pos};

    if (any_tensor_on_mps(color, rast, pos, tri, dy)) mps_sync();

    auto color_ref = tensor_to_mtl_buffer(color_c);
    auto rast_ref = tensor_to_mtl_buffer(rast_c);
    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto pos_ref = tensor_to_mtl_buffer(pos_c);
    auto dy_ref = tensor_to_mtl_buffer(dy_c);
    auto gc_ref = tensor_to_mtl_buffer(grad_color);
    auto gp_ref = tensor_to_mtl_buffer(grad_pos);
    auto work_ref = tensor_to_mtl_buffer(work_c);

    bool on_mps = grad_color.device().is_mps();
    auto pso = mtl_get_pipeline("AntialiasGradKernel");
    dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
        (void)cmdBuf;
        [enc setComputePipelineState:pso];
        [enc setBuffer:color_ref.buffer offset:color_ref.offset atIndex:0];
        [enc setBuffer:rast_ref.buffer  offset:rast_ref.offset  atIndex:1];
        [enc setBuffer:tri_ref.buffer   offset:tri_ref.offset   atIndex:2];
        [enc setBuffer:pos_ref.buffer   offset:pos_ref.offset   atIndex:3];
        [enc setBuffer:dy_ref.buffer    offset:dy_ref.offset    atIndex:4];
        [enc setBuffer:gc_ref.buffer    offset:gc_ref.offset    atIndex:5];
        [enc setBuffer:gp_ref.buffer    offset:gp_ref.offset    atIndex:6];
        [enc setBuffer:work_ref.buffer  offset:work_ref.offset  atIndex:7];
        [enc setBytes:&p length:sizeof(p) atIndex:8];

        NSUInteger threadGroupSize = MIN((NSUInteger)AA_GRAD_KERNEL_THREADS_PER_BLOCK,
                                         pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)workCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    });

    return {grad_color, grad_pos};
}

} // namespace mtldiffrast
