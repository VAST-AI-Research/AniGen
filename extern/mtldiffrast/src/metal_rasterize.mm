// Metal rasterization implementation — zero-copy MPS path.
// Uses hardware render pipeline for rasterization.
// Falls back to compute kernel only when render pipeline is unavailable.
#import "metal_rasterize.h"
#import "metal_utils.h"

namespace mtldiffrast {

MtlRasterizeContext::MtlRasterizeContext() {
    device_ = mtl_get_device();
    queue_ = mtl_get_queue();
    library_ = mtl_get_library();
    setup_render_pipeline();
    setup_compute_pipeline();
}

MtlRasterizeContext::~MtlRasterizeContext() {}

void MtlRasterizeContext::setup_compute_pipeline() {
    NSError* error = nil;
    id<MTLFunction> func = [library_ newFunctionWithName:@"rasterize_compute_kernel"];
    if (func) {
        compute_pipeline_ = [device_ newComputePipelineStateWithFunction:func error:&error];
    }
}

void MtlRasterizeContext::setup_render_pipeline() {
    NSError* error = nil;

    id<MTLFunction> vertexFunc = [library_ newFunctionWithName:@"rasterize_vertex"];
    id<MTLFunction> fragmentFunc = [library_ newFunctionWithName:@"rasterize_fragment"];
    if (!vertexFunc || !fragmentFunc) return;

    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vertexFunc;
    desc.fragmentFunction = fragmentFunc;

    // Color attachment 0: RGBA32Float — stores (u, v, z/w, tri_id+1)
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    desc.colorAttachments[0].blendingEnabled = NO;

    // Depth format for hardware depth test
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    // Rasterization: front-face only (CCW winding matches edge function at>0)
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

    render_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:desc error:&error];
    TORCH_CHECK(render_pipeline_ != nil, "[mtldiffrast] Failed to create render pipeline: ",
                error ? [[error localizedDescription] UTF8String] : "unknown");

    // Depth stencil state: Less comparison — matches nvdiffrast's cudaraster,
    // which keeps the SMALLEST z/w (atomicMin + "depth > oldDepth -> zkill").
    // Strict Less (not LessEqual) makes equal-depth fragments keep-first
    // (draw/triangle order), matching nvdiffrast's deterministic tie-break.
    MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
    dsDesc.depthCompareFunction = MTLCompareFunctionLess;
    dsDesc.depthWriteEnabled = YES;
    depth_stencil_ = [device_ newDepthStencilStateWithDescriptor:dsDesc];
}

std::tuple<torch::Tensor, torch::Tensor> MtlRasterizeContext::rasterize(
    const torch::Tensor& pos,
    const torch::Tensor& tri,
    const std::vector<int>& resolution
) {
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 4, "pos must be [V, 4]");
    TORCH_CHECK(tri.dim() == 2 && tri.size(1) == 3, "tri must be [T, 3]");
    TORCH_CHECK(resolution.size() == 2, "resolution must be [H, W]");

    int H = resolution[0];
    int W = resolution[1];
    int num_triangles = tri.size(0);
    int num_vertices = pos.size(0);

    auto pos_c = pos.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32);

    // Output tensors follow the input's device. On MPS they're MPS-backed
    // via _safe_zeros_on_device; render-pipeline writes reach them through a
    // blit encoder below (no more MTLTexture getBytes, no CPU round-trip).
    bool on_mps = tensor_is_mps(pos);
    auto rast_out = make_output_tensor({H, W, 4}, torch::kFloat32, pos);
    auto rast_db  = make_output_tensor({H, W, 4}, torch::kFloat32, pos);

    auto pos_ref = tensor_to_mtl_buffer(pos_c);
    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto out_ref = tensor_to_mtl_buffer(rast_out);
    auto db_ref = tensor_to_mtl_buffer(rast_db);

    if (render_pipeline_) {
        // ─── Hardware render pipeline path ───────────────────────────
        // One cmdBuf owns three encoders in order:
        //   1) render encoder → draws into colorTex (Shared storage),
        //   2) blit encoder   → copies colorTex → out_ref.buffer (rast_out),
        //   3) compute encoder → reads rast_out, writes rast_db derivatives.
        // On MPS the whole sequence encodes into the active MPSStream; on CPU
        // dispatch_batch commits+waits so the tensor data is valid on return.

        // Create standalone shared render target texture
        MTLTextureDescriptor* colorTexDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                        width:W height:H mipmapped:NO];
        colorTexDesc.usage = MTLTextureUsageRenderTarget;
        colorTexDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> colorTex = [device_ newTextureWithDescriptor:colorTexDesc];

        // Depth texture (transient — GPU only)
        MTLTextureDescriptor* depthTexDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                        width:W height:H mipmapped:NO];
        depthTexDesc.usage = MTLTextureUsageRenderTarget;
        depthTexDesc.storageMode = MTLStorageModePrivate;
        id<MTLTexture> depthTex = [device_ newTextureWithDescriptor:depthTexDesc];

        // Render pass descriptor
        MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = colorTex;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

        rpd.depthAttachment.texture = depthTex;
        rpd.depthAttachment.loadAction = MTLLoadActionClear;
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        rpd.depthAttachment.clearDepth = 1.0;  // far plane; first fragment (depth<1) passes Less

        struct {
            int num_triangles;
            int num_vertices;
            int width;
            int height;
            float xs, xo, ys, yo;
            int chunk_start;
        } db_params;
        db_params.num_triangles = num_triangles;
        db_params.num_vertices = num_vertices;
        db_params.width = W;
        db_params.height = H;
        db_params.xs = 2.0f / (float)W;
        db_params.xo = -1.0f + 1.0f / (float)W;
        db_params.ys = 2.0f / (float)H;
        db_params.yo = -1.0f + 1.0f / (float)H;
        db_params.chunk_start = 0;

        auto db_pso = mtl_get_pipeline("rasterize_db_kernel");
        auto render_pso = render_pipeline_;
        auto depth_state = depth_stencil_;
        int nverts = num_vertices;
        NSUInteger bytesPerRow = W * 4 * sizeof(float);

        dispatch_batch(on_mps, ^(id<MTLCommandBuffer> cmdBuf) {
            // (1) render encoder
            id<MTLRenderCommandEncoder> enc = [cmdBuf renderCommandEncoderWithDescriptor:rpd];
            [enc setRenderPipelineState:render_pso];
            [enc setDepthStencilState:depth_state];
            [enc setFrontFacingWinding:MTLWindingCounterClockwise];
            [enc setCullMode:MTLCullModeNone];
            // Flip Y to match compute kernel convention (Y=0 at bottom)
            MTLViewport viewport = {0.0, (double)H, (double)W, -(double)H, 0.0, 1.0};
            [enc setViewport:viewport];
            [enc setVertexBuffer:pos_ref.buffer offset:pos_ref.offset atIndex:0];
            [enc setVertexBuffer:tri_ref.buffer offset:tri_ref.offset atIndex:1];
            [enc setVertexBytes:&nverts length:sizeof(int) atIndex:2];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:num_triangles * 3];
            [enc endEncoding];

            // (2) blit: colorTex -> rast_out buffer. Metal guarantees
            // encoder ordering within a cmdBuf, so the compute pass below
            // sees the blit's writes to the rast_out buffer.
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromTexture:colorTex
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(W, H, 1)
                         toBuffer:out_ref.buffer
                destinationOffset:out_ref.offset
           destinationBytesPerRow:bytesPerRow
         destinationBytesPerImage:bytesPerRow * H];
            [blit endEncoding];

            // (3) compute: rast_db derivatives from rast_out and pos/tri
            id<MTLComputeCommandEncoder> dbEnc = [cmdBuf computeCommandEncoder];
            [dbEnc setComputePipelineState:db_pso];
            [dbEnc setBuffer:pos_ref.buffer offset:pos_ref.offset atIndex:0];
            [dbEnc setBuffer:tri_ref.buffer offset:tri_ref.offset atIndex:1];
            [dbEnc setBytes:&db_params length:sizeof(db_params) atIndex:2];
            [dbEnc setBuffer:out_ref.buffer offset:out_ref.offset atIndex:3];
            [dbEnc setBuffer:db_ref.buffer offset:db_ref.offset atIndex:4];
            [dbEnc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [dbEnc endEncoding];
        });

    } else {
        // ─── Compute kernel fallback ─────────────────────────────────
        struct {
            int num_triangles;
            int num_vertices;
            int width;
            int height;
            float xs, xo, ys, yo;
            int chunk_start;
        } params;

        params.num_vertices = num_vertices;
        params.width = W;
        params.height = H;
        params.xs = 2.0f / (float)W;
        params.xo = -1.0f + 1.0f / (float)W;
        params.ys = 2.0f / (float)H;
        params.yo = -1.0f + 1.0f / (float)H;

        const int CHUNK_SIZE = 100000;
        auto compute_pso = compute_pipeline_;

        if (num_triangles <= CHUNK_SIZE) {
            params.num_triangles = num_triangles;
            params.chunk_start = 0;
            dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
                (void)cmdBuf;
                [enc setComputePipelineState:compute_pso];
                [enc setBuffer:pos_ref.buffer offset:pos_ref.offset atIndex:0];
                [enc setBuffer:tri_ref.buffer offset:tri_ref.offset atIndex:1];
                [enc setBytes:&params length:sizeof(params) atIndex:2];
                [enc setBuffer:out_ref.buffer offset:out_ref.offset atIndex:3];
                [enc setBuffer:db_ref.buffer offset:db_ref.offset atIndex:4];
                [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            });
        } else {
            for (int chunk_start = 0; chunk_start < num_triangles; chunk_start += CHUNK_SIZE) {
                int chunk_end = std::min(chunk_start + CHUNK_SIZE, num_triangles);
                params.num_triangles = chunk_end - chunk_start;
                params.chunk_start = chunk_start;

                NSUInteger tri_byte_offset = tri_ref.offset + chunk_start * 3 * sizeof(int);

                dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
                    (void)cmdBuf;
                    [enc setComputePipelineState:compute_pso];
                    [enc setBuffer:pos_ref.buffer offset:pos_ref.offset atIndex:0];
                    [enc setBuffer:tri_ref.buffer offset:tri_byte_offset atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    [enc setBuffer:out_ref.buffer offset:out_ref.offset atIndex:3];
                    [enc setBuffer:db_ref.buffer offset:db_ref.offset atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
                });
            }
        }
    }

    return {rast_out, rast_db};
}

torch::Tensor MtlRasterizeContext::rasterize_grad(
    const torch::Tensor& pos,
    const torch::Tensor& tri,
    const torch::Tensor& rast_out,
    const torch::Tensor& dy,
    const torch::Tensor& ddb
) {
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 4, "pos must be [V, 4]");
    TORCH_CHECK(tri.dim() == 2 && tri.size(1) == 3, "tri must be [T, 3]");

    int V = pos.size(0);
    int T = tri.size(0);
    int H = rast_out.size(0);
    int W = rast_out.size(1);

    auto pos_c = pos.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32).view({-1});
    auto rast_c = rast_out.contiguous().to(torch::kFloat32);
    auto dy_c = dy.contiguous().to(torch::kFloat32);
    bool enable_db = ddb.defined() && ddb.numel() > 0;
    auto ddb_c = enable_db ? ddb.contiguous().to(torch::kFloat32) : torch::zeros({1}, torch::kFloat32);

    auto grad_pos = make_output_tensor({V, 4}, torch::kFloat32, pos);

    struct {
        int num_triangles;
        int num_vertices;
        int width;
        int height;
        float xs, xo, ys, yo;
        int enable_db;
    } params;
    params.num_triangles = T;
    params.num_vertices = V;
    params.width = W;
    params.height = H;
    params.xs = 2.0f / (float)W;
    params.xo = -1.0f + 1.0f / (float)W;
    params.ys = 2.0f / (float)H;
    params.yo = -1.0f + 1.0f / (float)H;
    params.enable_db = enable_db ? 1 : 0;

    // No mps_sync: dispatch_kernel on MPS encodes into the active MPSStream.

    auto pos_ref = tensor_to_mtl_buffer(pos_c);
    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto rast_ref = tensor_to_mtl_buffer(rast_c);
    auto dy_ref = tensor_to_mtl_buffer(dy_c);
    auto ddb_ref = tensor_to_mtl_buffer(ddb_c);
    auto grad_ref = tensor_to_mtl_buffer(grad_pos);

    auto pso = mtl_get_pipeline("rasterize_grad_kernel");

    bool on_mps = tensor_is_mps(pos);
    dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
        (void)cmdBuf;
        [enc setComputePipelineState:pso];
        [enc setBuffer:pos_ref.buffer  offset:pos_ref.offset  atIndex:0];
        [enc setBuffer:tri_ref.buffer  offset:tri_ref.offset  atIndex:1];
        [enc setBuffer:rast_ref.buffer offset:rast_ref.offset atIndex:2];
        [enc setBuffer:dy_ref.buffer   offset:dy_ref.offset   atIndex:3];
        [enc setBuffer:ddb_ref.buffer  offset:ddb_ref.offset  atIndex:4];
        [enc setBuffer:grad_ref.buffer offset:grad_ref.offset atIndex:5];
        [enc setBytes:&params length:sizeof(params) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    });

    return grad_pos;
}

} // namespace mtldiffrast
