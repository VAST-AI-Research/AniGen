// Metal texture implementation — forward/backward pass host-side dispatch.
// Zero-copy MPS path via metal_utils.h.
#import "metal_texture.h"
#import "metal_utils.h"

namespace mtldiffrast {

//------------------------------------------------------------------------
// Kernel params struct — must match texture.metal TextureKernelParams exactly.

struct TextureKernelParams
{
    int texOfs[TEX_MAX_MIP_LEVEL];
    int uvOfs;
    int uvDAOfs;
    int mipLevelBiasOfs;
    int outOfs;
    int enableMip;
    int filterMode;
    int boundaryMode;
    int texConst;
    int mipLevelLimit;
    int channels;
    int imgWidth;
    int imgHeight;
    int texWidth;
    int texHeight;
    int texDepth;
    int n;
    int mipLevelMax;
    int mipLevelOut;
};

//------------------------------------------------------------------------
// Mip info calculation.

struct MipSize { int x; int y; };

static MipSize mipLevelSizeCPU(int texWidth, int texHeight, int i)
{
    int w = (texWidth  >> i) > 1 ? (texWidth  >> i) : 1;
    int h = (texHeight >> i) > 1 ? (texHeight >> i) : 1;
    return {w, h};
}

static int calculateMipInfo(int texWidth, int texHeight, int texDepth, int channels,
                             int boundaryMode, int mipLevelLimit,
                             int* mipOffsets, int& mipLevelMax)
{
    if (mipLevelLimit == 0) {
        mipLevelMax = 0;
        return 0;
    }

    int w = texWidth;
    int h = texHeight;
    int mipTotal = 0;
    int level = 0;
    int c = (boundaryMode == TEX_BOUNDARY_MODE_CUBE) ? (channels * 6) : channels;
    mipOffsets[0] = 0;

    while ((w | h) > 1) {
        level += 1;
        TORCH_CHECK(!((w > 1 && (w & 1)) || (h > 1 && (h & 1))),
            "Cannot build mip chain: odd dimension at level ", level,
            " (", w, "x", h, "). Resize to power-of-two.");
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;
        mipOffsets[level] = mipTotal;
        mipTotal += w * h * texDepth * c;
        if (mipLevelLimit >= 0 && level == mipLevelLimit)
            break;
    }

    mipLevelMax = level;
    return mipTotal;
}

//------------------------------------------------------------------------
// Mipmap construction.

TextureMipWrapper texture_construct_mip(const torch::Tensor& tex, int max_mip_level, bool cube_mode)
{
    TORCH_CHECK(tex.is_contiguous(), "tex must be contiguous");
    TORCH_CHECK(tex.scalar_type() == torch::kFloat32, "tex must be float32");

    int texDepth, texHeight, texWidth, channels;
    if (!cube_mode) {
        TORCH_CHECK(tex.dim() == 4, "tex must be [D, H, W, C]");
        texDepth   = tex.size(0);
        texHeight  = tex.size(1);
        texWidth   = tex.size(2);
        channels   = tex.size(3);
    } else {
        TORCH_CHECK(tex.dim() == 5 && tex.size(1) == 6, "tex must be [D, 6, H, W, C] in cube mode");
        TORCH_CHECK(tex.size(2) == tex.size(3), "cube map must be square");
        texDepth   = tex.size(0);
        texHeight  = tex.size(2);
        texWidth   = tex.size(3);
        channels   = tex.size(4);
    }

    int boundaryMode = cube_mode ? TEX_BOUNDARY_MODE_CUBE : TEX_BOUNDARY_MODE_WRAP;
    // Heap-allocated so the dispatch_batch block can capture the pointer.
    std::vector<int> mipOffsetsVec(TEX_MAX_MIP_LEVEL, 0);
    int* mipOffsets = mipOffsetsVec.data();
    int mipLevelMax = 0;
    int mipTotal = calculateMipInfo(texWidth, texHeight, texDepth, channels,
                                     boundaryMode, max_mip_level, mipOffsets, mipLevelMax);

    torch::Tensor mip = torch::empty({mipTotal}, torch::TensorOptions().dtype(torch::kFloat32));

    int channelDivIdx = 0;
    if (!(channels & 3))      channelDivIdx = 2;
    else if (!(channels & 1)) channelDivIdx = 1;
    const char* kernelNames[3] = {"MipBuildKernel1", "MipBuildKernel2", "MipBuildKernel4"};

    auto dev = mtl_get_device();
    auto queue = mtl_get_queue();
    auto pipeline = mtl_get_pipeline(kernelNames[channelDivIdx]);

    auto texFlat = tex.contiguous().view({-1});

    // mip is CPU-allocated (torch::empty without device), and texFlat shares
    // storage with `tex`. If `tex` is MPS the caller is expected to stage to
    // CPU beforehand — mip construction is setup-time CPU work. dispatch_batch
    // with on_mps=false wraps the entire mip-build chain in a single cmdBuf
    // while keeping the endKernelCoalescing + autorelease pattern consistent
    // with the rest of the file.
    id<MTLBuffer> mipBuf = [dev newBufferWithBytesNoCopy:mip.data_ptr<float>()
                                                  length:mipTotal * sizeof(float)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];

    id<MTLBuffer> texBuf = [dev newBufferWithBytesNoCopy:texFlat.data_ptr<float>()
                                                  length:texFlat.numel() * sizeof(float)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];

    (void)queue;
    dispatch_batch(/*on_mps=*/false, ^(id<MTLCommandBuffer> cmdBuf) {
        for (int i = 1; i <= mipLevelMax; i++)
        {
            auto ms = mipLevelSizeCPU(texWidth, texHeight, i);
            int outW = ms.x;
            int outH = ms.y;
            int depth = texDepth * (cube_mode ? 6 : 1);

            TextureKernelParams kp = {};
            kp.channels = channels;
            kp.texWidth = texWidth;
            kp.texHeight = texHeight;
            kp.mipLevelOut = i;

            id<MTLBuffer> inBuf;
            int inOffset = 0;
            if (i == 1) {
                inBuf = texBuf;
                inOffset = 0;
            } else {
                inBuf = mipBuf;
                inOffset = mipOffsets[i - 1] * sizeof(float);
            }

            int outOffset = mipOffsets[i] * sizeof(float);

            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:inBuf  offset:inOffset  atIndex:0];
            [enc setBuffer:mipBuf offset:outOffset atIndex:1];
            [enc setBytes:&kp length:sizeof(kp) atIndex:2];

            MTLSize gridSize = MTLSizeMake(outW, outH, depth);
            NSUInteger tw = pipeline.threadExecutionWidth;
            NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
            if (th > (NSUInteger)outH) th = (NSUInteger)outH;
            MTLSize threadgroupSize = MTLSizeMake(tw, th, 1);

            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];
        }
    });

    TextureMipWrapper wrapper;
    wrapper.mip = mip;
    wrapper.max_mip_level = max_mip_level;
    wrapper.texture_size = tex.sizes().vec();
    wrapper.cube_mode = cube_mode;
    return wrapper;
}

//------------------------------------------------------------------------
// Forward texture lookup.

torch::Tensor texture_fwd_mip(
    const torch::Tensor& tex,
    const torch::Tensor& uv,
    const torch::Tensor& uv_da,
    const torch::Tensor& mip_level_bias,
    TextureMipWrapper& mip_wrapper,
    const std::vector<torch::Tensor>& mip_stack,
    int filter_mode,
    int boundary_mode
)
{
    TORCH_CHECK(tex.is_contiguous() && tex.scalar_type() == torch::kFloat32, "tex must be contiguous float32");
    TORCH_CHECK(uv.is_contiguous() && uv.scalar_type() == torch::kFloat32, "uv must be contiguous float32");

    bool cube_mode = (boundary_mode == TEX_BOUNDARY_MODE_CUBE);
    bool has_mip_stack = (mip_stack.size() > 0);
    bool enableMip = (filter_mode == TEX_MODE_LINEAR_MIPMAP_NEAREST || filter_mode == TEX_MODE_LINEAR_MIPMAP_LINEAR);
    bool has_uv_da = uv_da.defined() && uv_da.numel() > 0;
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.numel() > 0;
    int max_mip_level = has_mip_stack ? (int)mip_stack.size() : mip_wrapper.max_mip_level;

    int texDepth, texHeight, texWidth, channels;
    if (!cube_mode) {
        TORCH_CHECK(tex.dim() == 4, "tex must be [D, H, W, C]");
        texDepth   = tex.size(0);
        texHeight  = tex.size(1);
        texWidth   = tex.size(2);
        channels   = tex.size(3);
    } else {
        TORCH_CHECK(tex.dim() == 5 && tex.size(1) == 6, "tex must be [D, 6, H, W, C] in cube mode");
        texDepth   = tex.size(0);
        texHeight  = tex.size(2);
        texWidth   = tex.size(3);
        channels   = tex.size(4);
    }

    TORCH_CHECK(uv.dim() == 4, "uv must be [N, H, W, 2|3]");
    int n         = uv.size(0);
    int imgHeight = uv.size(1);
    int imgWidth  = uv.size(2);

    TextureKernelParams kp = {};
    kp.filterMode     = filter_mode;
    kp.boundaryMode   = boundary_mode;
    kp.enableMip      = enableMip ? 1 : 0;
    kp.channels       = channels;
    kp.imgWidth       = imgWidth;
    kp.imgHeight      = imgHeight;
    kp.texWidth       = texWidth;
    kp.texHeight      = texHeight;
    kp.texDepth       = texDepth;
    kp.n              = n;
    kp.mipLevelLimit  = enableMip ? max_mip_level : 0;
    kp.uvOfs          = 0;
    kp.outOfs         = 0;
    kp.uvDAOfs        = (enableMip && has_uv_da) ? 0 : -1;
    kp.mipLevelBiasOfs = (enableMip && has_mip_level_bias) ? 0 : -1;

    int baseSizeFloats = (int)tex.numel();
    kp.texOfs[0] = 0;

    int mipOffsets[TEX_MAX_MIP_LEVEL] = {};
    if (enableMip)
    {
        if (has_mip_stack)
        {
            kp.mipLevelMax = max_mip_level;
            int ofs = baseSizeFloats;
            for (int i = 1; i <= kp.mipLevelMax; i++)
            {
                kp.texOfs[i] = ofs;
                ofs += (int)mip_stack[i - 1].numel();
            }
        }
        else
        {
            int mipLevelMax = 0;
            int mipTotal = calculateMipInfo(texWidth, texHeight, texDepth, channels,
                                             boundary_mode, max_mip_level, mipOffsets, mipLevelMax);
            kp.mipLevelMax = mipLevelMax;
            for (int i = 1; i <= mipLevelMax; i++)
                kp.texOfs[i] = baseSizeFloats + mipOffsets[i];
        }
    }

    // Output on same device as input
    torch::Tensor out = make_empty_tensor({n, imgHeight, imgWidth, channels}, torch::kFloat32, tex);

    // Build combined texture buffer
    torch::Tensor texCombined;
    if (enableMip && kp.mipLevelMax > 0)
    {
        if (has_mip_stack)
        {
            std::vector<torch::Tensor> parts;
            parts.push_back(tex.contiguous().view({-1}));
            for (int i = 0; i < (int)mip_stack.size(); i++)
                parts.push_back(mip_stack[i].contiguous().view({-1}));
            texCombined = torch::cat(parts, 0);
        }
        else
        {
            texCombined = torch::cat({tex.contiguous().view({-1}), mip_wrapper.mip.contiguous()}, 0);
        }
    }
    else
    {
        texCombined = tex.contiguous().view({-1});
    }

    int channelDivIdx = 0;
    if (!(channels & 3))      channelDivIdx = 2;
    else if (!(channels & 1)) channelDivIdx = 1;
    const char* kernelNames[3] = {"TextureFwdKernel1", "TextureFwdKernel2", "TextureFwdKernel4"};

    auto pipeline = mtl_get_pipeline(kernelNames[channelDivIdx]);

    // No mps_sync: dispatch_kernel on MPS encodes into the active MPSStream,
    // which handles ordering with surrounding torch.mps work. CPU path is
    // synchronous via commit+wait inside dispatch_kernel.

    auto tex_ref = tensor_to_mtl_buffer(texCombined);
    auto uvContig = uv.contiguous();
    auto uv_ref = tensor_to_mtl_buffer(uvContig);

    MtlBufferRef uvda_ref;
    torch::Tensor uvDAContig;
    if (enableMip && has_uv_da) {
        uvDAContig = uv_da.contiguous();
        uvda_ref = tensor_to_mtl_buffer(uvDAContig);
    } else {
        static auto dummy = torch::zeros({1}, torch::kFloat32);
        uvda_ref = tensor_to_mtl_buffer(dummy);
    }

    MtlBufferRef mipbias_ref;
    torch::Tensor mipBiasContig;
    if (enableMip && has_mip_level_bias) {
        mipBiasContig = mip_level_bias.contiguous();
        mipbias_ref = tensor_to_mtl_buffer(mipBiasContig);
    } else {
        static auto dummy = torch::zeros({1}, torch::kFloat32);
        mipbias_ref = tensor_to_mtl_buffer(dummy);
    }

    auto out_ref = tensor_to_mtl_buffer(out);

    MTLSize gridSize = MTLSizeMake(imgWidth, imgHeight, n);
    NSUInteger tw = pipeline.threadExecutionWidth;
    NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
    if (th > (NSUInteger)imgHeight) th = (NSUInteger)imgHeight;
    MTLSize threadgroupSize = MTLSizeMake(tw, th, 1);

    bool on_mps = tensor_is_mps(tex);
    dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
        (void)cmdBuf;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:tex_ref.buffer     offset:tex_ref.offset     atIndex:0];
        [enc setBuffer:uv_ref.buffer      offset:uv_ref.offset      atIndex:1];
        [enc setBuffer:uvda_ref.buffer    offset:uvda_ref.offset    atIndex:2];
        [enc setBuffer:mipbias_ref.buffer offset:mipbias_ref.offset atIndex:3];
        [enc setBuffer:out_ref.buffer     offset:out_ref.offset     atIndex:4];
        [enc setBytes:&kp length:sizeof(kp) atIndex:5];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    });

    return out;
}

//------------------------------------------------------------------------
// Convenience: forward without mips.

torch::Tensor texture_fwd(
    const torch::Tensor& tex,
    const torch::Tensor& uv,
    int filter_mode,
    int boundary_mode
)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    TextureMipWrapper empty_wrapper;
    return texture_fwd_mip(tex, uv, empty_tensor, empty_tensor, empty_wrapper, empty_vector, filter_mode, boundary_mode);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> texture_grad(
    const torch::Tensor& tex,
    const torch::Tensor& uv,
    const torch::Tensor& dy,
    const torch::Tensor& uv_da,
    const torch::Tensor& mip_level_bias,
    TextureMipWrapper& mip_wrapper,
    const std::vector<torch::Tensor>& mip_stack,
    int filter_mode,
    int boundary_mode
) {
    TORCH_CHECK(tex.is_contiguous() && tex.scalar_type() == torch::kFloat32);
    TORCH_CHECK(uv.is_contiguous() && uv.scalar_type() == torch::kFloat32);
    TORCH_CHECK(dy.is_contiguous() && dy.scalar_type() == torch::kFloat32);

    bool cube_mode = (boundary_mode == TEX_BOUNDARY_MODE_CUBE);
    bool has_mip_stack = (mip_stack.size() > 0);
    bool enableMip = (filter_mode == TEX_MODE_LINEAR_MIPMAP_NEAREST || filter_mode == TEX_MODE_LINEAR_MIPMAP_LINEAR);
    bool has_uv_da = uv_da.defined() && uv_da.numel() > 0;
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.numel() > 0;
    int max_mip_level = has_mip_stack ? (int)mip_stack.size() : mip_wrapper.max_mip_level;

    int texDepth, texHeight, texWidth, channels;
    if (!cube_mode) {
        texDepth = tex.size(0); texHeight = tex.size(1); texWidth = tex.size(2); channels = tex.size(3);
    } else {
        texDepth = tex.size(0); texHeight = tex.size(2); texWidth = tex.size(3); channels = tex.size(4);
    }

    int n = uv.size(0), imgHeight = uv.size(1), imgWidth = uv.size(2);

    TextureKernelParams kp = {};
    kp.filterMode = filter_mode;
    kp.boundaryMode = boundary_mode;
    kp.enableMip = enableMip ? 1 : 0;
    kp.channels = channels;
    kp.imgWidth = imgWidth;
    kp.imgHeight = imgHeight;
    kp.texWidth = texWidth;
    kp.texHeight = texHeight;
    kp.texDepth = texDepth;
    kp.n = n;
    kp.mipLevelLimit = enableMip ? max_mip_level : 0;
    kp.uvOfs = 0;
    kp.outOfs = 0;
    kp.uvDAOfs = (enableMip && has_uv_da) ? 0 : -1;
    kp.mipLevelBiasOfs = (enableMip && has_mip_level_bias) ? 0 : -1;

    int baseSizeFloats = (int)tex.numel();
    kp.texOfs[0] = 0;

    int mipOffsets[TEX_MAX_MIP_LEVEL] = {};
    if (enableMip) {
        if (has_mip_stack) {
            kp.mipLevelMax = max_mip_level;
            int ofs = baseSizeFloats;
            for (int i = 1; i <= kp.mipLevelMax; i++) {
                kp.texOfs[i] = ofs;
                ofs += (int)mip_stack[i - 1].numel();
            }
        } else {
            int mipLevelMax = 0;
            int mipTotal = calculateMipInfo(texWidth, texHeight, texDepth, channels,
                                             boundary_mode, max_mip_level, mipOffsets, mipLevelMax);
            kp.mipLevelMax = mipLevelMax;
            for (int i = 1; i <= mipLevelMax; i++)
                kp.texOfs[i] = baseSizeFloats + mipOffsets[i];
        }
    }

    // Build combined texture buffer.
    torch::Tensor texCombined;
    if (enableMip && kp.mipLevelMax > 0) {
        if (has_mip_stack) {
            std::vector<torch::Tensor> parts;
            parts.push_back(tex.contiguous().view({-1}));
            for (int i = 0; i < (int)mip_stack.size(); i++)
                parts.push_back(mip_stack[i].contiguous().view({-1}));
            texCombined = torch::cat(parts, 0);
        } else {
            texCombined = torch::cat({tex.contiguous().view({-1}), mip_wrapper.mip.contiguous()}, 0);
        }
    } else {
        texCombined = tex.contiguous().view({-1});
    }

    // Gradient outputs on same device as input
    auto gradTex = torch::zeros_like(texCombined);
    auto gradUV = make_output_tensor({n, imgHeight, imgWidth, cube_mode ? 3 : 2}, torch::kFloat32, tex);
    torch::Tensor gradUVDA, gradMipBias;
    bool need_uvda_grad = (filter_mode == TEX_MODE_LINEAR_MIPMAP_LINEAR && has_uv_da);
    bool need_mip_grad = (filter_mode == TEX_MODE_LINEAR_MIPMAP_LINEAR && has_mip_level_bias);
    if (need_uvda_grad)
        gradUVDA = make_output_tensor({n, imgHeight, imgWidth, cube_mode ? 6 : 4}, torch::kFloat32, tex);
    else
        gradUVDA = torch::zeros({1}, torch::kFloat32);
    if (need_mip_grad)
        gradMipBias = make_output_tensor({n, imgHeight, imgWidth}, torch::kFloat32, tex);
    else
        gradMipBias = torch::zeros({1}, torch::kFloat32);

    // No mps_sync: dispatch_kernel on MPS encodes into the active MPSStream.

    auto tex_ref = tensor_to_mtl_buffer(texCombined);
    auto uvContig = uv.contiguous();
    auto uv_ref = tensor_to_mtl_buffer(uvContig);

    MtlBufferRef uvda_ref;
    torch::Tensor uvDAContig;
    if (enableMip && has_uv_da) {
        uvDAContig = uv_da.contiguous();
        uvda_ref = tensor_to_mtl_buffer(uvDAContig);
    } else {
        static auto dummy = torch::zeros({1}, torch::kFloat32);
        uvda_ref = tensor_to_mtl_buffer(dummy);
    }

    MtlBufferRef mipbias_ref;
    torch::Tensor mipBiasContig;
    if (enableMip && has_mip_level_bias) {
        mipBiasContig = mip_level_bias.contiguous();
        mipbias_ref = tensor_to_mtl_buffer(mipBiasContig);
    } else {
        static auto dummy = torch::zeros({1}, torch::kFloat32);
        mipbias_ref = tensor_to_mtl_buffer(dummy);
    }

    auto dyContig = dy.contiguous();
    auto dy_ref = tensor_to_mtl_buffer(dyContig);
    auto gt_ref = tensor_to_mtl_buffer(gradTex);
    auto gu_ref = tensor_to_mtl_buffer(gradUV);
    auto guda_ref = tensor_to_mtl_buffer(gradUVDA);
    auto gm_ref = tensor_to_mtl_buffer(gradMipBias);

    auto pipeline = mtl_get_pipeline("TextureGradKernel");

    MTLSize gridSize = MTLSizeMake(imgWidth, imgHeight, n);
    NSUInteger tw = pipeline.threadExecutionWidth;
    NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
    if (th > (NSUInteger)imgHeight) th = (NSUInteger)imgHeight;
    MTLSize tgSize = MTLSizeMake(tw, th, 1);

    bool on_mps = tensor_is_mps(tex);

    dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
        (void)cmdBuf;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:tex_ref.buffer     offset:tex_ref.offset     atIndex:0];
        [enc setBuffer:uv_ref.buffer      offset:uv_ref.offset      atIndex:1];
        [enc setBuffer:uvda_ref.buffer    offset:uvda_ref.offset    atIndex:2];
        [enc setBuffer:mipbias_ref.buffer offset:mipbias_ref.offset atIndex:3];
        [enc setBuffer:dy_ref.buffer      offset:dy_ref.offset      atIndex:4];
        [enc setBuffer:gt_ref.buffer      offset:gt_ref.offset      atIndex:5];
        [enc setBuffer:gu_ref.buffer      offset:gu_ref.offset      atIndex:6];
        [enc setBuffer:guda_ref.buffer    offset:guda_ref.offset    atIndex:7];
        [enc setBuffer:gm_ref.buffer      offset:gm_ref.offset      atIndex:8];
        [enc setBytes:&kp length:sizeof(kp) atIndex:9];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    });

    // Mip grad kernel — read-modify-write on gradTex; runs after the main
    // grad kernel (same queue ordering on both MPS and CPU paths).
    if (enableMip && kp.mipLevelMax > 0) {
        auto mipPipeline = mtl_get_pipeline("MipGradKernel");
        int depth = texDepth * (cube_mode ? 6 : 1);

        MTLSize mipGrid = MTLSizeMake(texWidth, texHeight, depth);
        NSUInteger mtw = mipPipeline.threadExecutionWidth;
        NSUInteger mth = mipPipeline.maxTotalThreadsPerThreadgroup / mtw;
        if (mth > (NSUInteger)texHeight) mth = (NSUInteger)texHeight;
        MTLSize mipTg = MTLSizeMake(mtw, mth, 1);

        dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
            (void)cmdBuf;
            [enc setComputePipelineState:mipPipeline];
            [enc setBuffer:gt_ref.buffer offset:gt_ref.offset atIndex:0];
            [enc setBytes:&kp length:sizeof(kp) atIndex:1];
            [enc dispatchThreads:mipGrid threadsPerThreadgroup:mipTg];
        });
    }

    auto gradTexOut = gradTex.slice(0, 0, baseSizeFloats).view(tex.sizes());

    return {gradTexOut, gradUV,
            need_uvda_grad ? gradUVDA : torch::Tensor(),
            need_mip_grad ? gradMipBias : torch::Tensor()};
}

} // namespace mtldiffrast
