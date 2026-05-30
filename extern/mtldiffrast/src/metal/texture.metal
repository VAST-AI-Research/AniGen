// Metal texture sampling kernels — forward + backward pass.
// Based on Laine et al. 2020, Modular Primitives for High-Performance Differentiable Rendering.
#include <metal_stdlib>
using namespace metal;

// Atomic float add via CAS loop for gradient accumulation.
inline void tex_atomicAddFloat(device atomic_uint* addr, float value)
{
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true)
    {
        float current = as_type<float>(expected);
        float desired = current + value;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired_bits,
                                                   memory_order_relaxed, memory_order_relaxed))
            break;
    }
}

//------------------------------------------------------------------------
// Constants for texture sampling modes.

#define TEX_MAX_MIP_LEVEL                       16
#define TEX_MODE_NEAREST                        0
#define TEX_MODE_LINEAR                         1
#define TEX_MODE_LINEAR_MIPMAP_NEAREST          2
#define TEX_MODE_LINEAR_MIPMAP_LINEAR           3
#define TEX_BOUNDARY_MODE_CUBE                  0
#define TEX_BOUNDARY_MODE_WRAP                  1
#define TEX_BOUNDARY_MODE_CLAMP                 2
#define TEX_BOUNDARY_MODE_ZERO                  3

//------------------------------------------------------------------------
// Kernel params struct for texture sampling operations.
// Pointers are passed as device buffer offsets; the kernel receives buffers
// separately and indexes via these offsets.

struct TextureKernelParams
{
    // Mip level base-pointer offsets into the flat texture buffer.
    // tex[0] = base texture offset, tex[1..N] = mip level offsets.
    int     texOfs[TEX_MAX_MIP_LEVEL];

    int     uvOfs;                  // Offset into uv buffer.
    int     uvDAOfs;                // Offset into uvDA buffer, or -1.
    int     mipLevelBiasOfs;        // Offset into mipLevelBias buffer, or -1.
    int     outOfs;                 // Offset into output buffer.

    int     enableMip;
    int     filterMode;
    int     boundaryMode;
    int     texConst;
    int     mipLevelLimit;
    int     channels;
    int     imgWidth;
    int     imgHeight;
    int     texWidth;
    int     texHeight;
    int     texDepth;
    int     n;
    int     mipLevelMax;
    int     mipLevelOut;            // For mip builder kernel.
};

//------------------------------------------------------------------------
// Helpers — mip level size.

inline int2 mipLevelSize(const constant TextureKernelParams& p, int i)
{
    int w = (p.texWidth  >> i) > 1 ? (p.texWidth  >> i) : 1;
    int h = (p.texHeight >> i) > 1 ? (p.texHeight >> i) : 1;
    return int2(w, h);
}

//------------------------------------------------------------------------
// Math helpers.

inline float lerp_f(float a, float b, float t)  { return a + t * (b - a); }
inline float2 lerp_f2(float2 a, float2 b, float t) { return a + t * (b - a); }
inline float4 lerp_f4(float4 a, float4 b, float t) { return a + t * (b - a); }

inline float bilerp_f(float a00, float a10, float a01, float a11, float2 uv)
{
    return lerp_f(lerp_f(a00, a10, uv.x), lerp_f(a01, a11, uv.x), uv.y);
}
inline float2 bilerp_f2(float2 a00, float2 a10, float2 a01, float2 a11, float2 uv)
{
    return lerp_f2(lerp_f2(a00, a10, uv.x), lerp_f2(a01, a11, uv.x), uv.y);
}
inline float4 bilerp_f4(float4 a00, float4 a10, float4 a01, float4 a11, float2 uv)
{
    return lerp_f4(lerp_f4(a00, a10, uv.x), lerp_f4(a01, a11, uv.x), uv.y);
}

//------------------------------------------------------------------------
// Cube map face adjacency tables — encodes wrap behavior at face boundaries.

constant uint32_t c_cubeWrapMask1[48] =
{
    0x1530a440, 0x1133a550, 0x6103a110, 0x1515aa44, 0x6161aa11, 0x40154a04, 0x44115a05, 0x04611a01,
    0x2630a440, 0x2233a550, 0x5203a110, 0x2626aa44, 0x5252aa11, 0x40264a04, 0x44225a05, 0x04521a01,
    0x32608064, 0x3366a055, 0x13062091, 0x32328866, 0x13132299, 0x50320846, 0x55330a55, 0x05130219,
    0x42508064, 0x4455a055, 0x14052091, 0x42428866, 0x14142299, 0x60420846, 0x66440a55, 0x06140219,
    0x5230a044, 0x5533a055, 0x1503a011, 0x5252aa44, 0x1515aa11, 0x40520a44, 0x44550a55, 0x04150a11,
    0x6130a044, 0x6633a055, 0x2603a011, 0x6161aa44, 0x2626aa11, 0x40610a44, 0x44660a55, 0x04260a11,
};

constant uint8_t c_cubeWrapMask2[48] =
{
    0x26, 0x33, 0x11, 0x05, 0x00, 0x09, 0x0c, 0x04, 0x04, 0x00, 0x00, 0x05, 0x00, 0x81, 0xc0, 0x40,
    0x02, 0x03, 0x09, 0x00, 0x0a, 0x00, 0x00, 0x02, 0x64, 0x30, 0x90, 0x55, 0xa0, 0x99, 0xcc, 0x64,
    0x24, 0x30, 0x10, 0x05, 0x00, 0x01, 0x00, 0x00, 0x06, 0x03, 0x01, 0x05, 0x00, 0x89, 0xcc, 0x44,
};

//------------------------------------------------------------------------
// wrapCubeMap — resolve out-of-bounds cube face coordinates via adjacency.

inline int4 wrapCubeMap(int face, int ix0, int ix1, int iy0, int iy1, int w)
{
    int cx = (ix0 < 0) ? 0 : (ix1 >= w) ? 2 : 1;
    int cy = (iy0 < 0) ? 0 : (iy1 >= w) ? 6 : 3;
    int c = cx + cy;
    if (c >= 5) c--;
    c = (face << 3) + c;

    uint32_t m = c_cubeWrapMask1[c];
    int x0 = (m >>  0) & 3; x0 = (x0 == 0) ? 0 : (x0 == 1) ? ix0 : iy0;
    int x1 = (m >>  2) & 3; x1 = (x1 == 0) ? 0 : (x1 == 1) ? ix1 : iy0;
    int x2 = (m >>  4) & 3; x2 = (x2 == 0) ? 0 : (x2 == 1) ? ix0 : iy1;
    int x3 = (m >>  6) & 3; x3 = (x3 == 0) ? 0 : (x3 == 1) ? ix1 : iy1;
    int y0 = (m >>  8) & 3; y0 = (y0 == 0) ? 0 : (y0 == 1) ? ix0 : iy0;
    int y1 = (m >> 10) & 3; y1 = (y1 == 0) ? 0 : (y1 == 1) ? ix1 : iy0;
    int y2 = (m >> 12) & 3; y2 = (y2 == 0) ? 0 : (y2 == 1) ? ix0 : iy1;
    int y3 = (m >> 14) & 3; y3 = (y3 == 0) ? 0 : (y3 == 1) ? ix1 : iy1;
    int f0 = ((m >> 16) & 15) - 1;
    int f1 = ((m >> 20) & 15) - 1;
    int f2 = ((m >> 24) & 15) - 1;
    int f3 = ((m >> 28)     ) - 1;

    uint32_t f = c_cubeWrapMask2[c];
    int w1 = w - 1;
    if (f & 0x01) x0 = w1 - x0;
    if (f & 0x02) x1 = w1 - x1;
    if (f & 0x04) x2 = w1 - x2;
    if (f & 0x08) x3 = w1 - x3;
    if (f & 0x10) y0 = w1 - y0;
    if (f & 0x20) y1 = w1 - y1;
    if (f & 0x40) y2 = w1 - y2;
    if (f & 0x80) y3 = w1 - y3;

    int4 tcOut;
    tcOut.x = x0 + (y0 + f0 * w) * w;
    tcOut.y = x1 + (y1 + f1 * w) * w;
    tcOut.z = x2 + (y2 + f2 * w) * w;
    tcOut.w = x3 + (y3 + f3 * w) * w;
    return tcOut;
}

//------------------------------------------------------------------------
// indexCubeMap — map 3D vector to (s,t) face coords + face index.
// Uses precise_rcp for reciprocal.

inline int indexCubeMap(thread float& x, thread float& y, float z)
{
    float ax = abs(x);
    float ay = abs(y);
    float az = abs(z);
    int idx;
    float c;
    if (az > max(ax, ay))       { idx = 4; c = z; }
    else if (ay > ax)           { idx = 2; c = y; y = z; }
    else                        { idx = 0; c = x; x = z; }
    if (c < 0.f) idx += 1;

    float m = 0.5f / abs(c);

    // Sign flip logic: bit (0x21u >> idx) & 1 controls sign of m0.
    float m0 = ((0x21u >> idx) & 1) ? -m : m;
    float m1 = (idx != 2) ? -m : m;
    x = x * m0 + 0.5f;
    y = y * m1 + 0.5f;
    if (!isfinite(x) || !isfinite(y))
        return -1;
    x = clamp(x, 0.f, 1.f);
    y = clamp(y, 0.f, 1.f);
    return idx;
}

//------------------------------------------------------------------------
// indexCubeMapGradST — compute d{s,t}/d{X,Y} from d{x,y,z}/d{X,Y}.

inline float4 indexCubeMapGradST(float3 uv, float3 dvdX, float3 dvdY)
{
    float ax = abs(uv.x);
    float ay = abs(uv.y);
    float az = abs(uv.z);
    int idx;
    float c, gu, gv;
    if (az > max(ax, ay))       { idx = 0x10; c = uv.z; gu = uv.x; gv = uv.y; }
    else if (ay > ax)           { idx = 0x04; c = uv.y; gu = uv.x; gv = uv.z; }
    else                        { idx = 0x01; c = uv.x; gu = uv.z; gv = uv.y; }
    if (c < 0.f) idx += idx;
    if (idx & 0x09)
    {
        dvdX.z = -dvdX.z;
        dvdY.z = -dvdY.z;
    }
    float m = 1.0f / abs(c);
    float dm = m * 0.5f;
    float mm = m * dm;
    gu *= (idx & 0x34) ? -mm : mm;
    gv *= (idx & 0x2e) ? -mm : mm;

    float4 res;
    if (idx & 0x03)
    {
        res = float4(gu * dvdX.x + dm * dvdX.z,
                     gu * dvdY.x + dm * dvdY.z,
                     gv * dvdX.x - dm * dvdX.y,
                     gv * dvdY.x - dm * dvdY.y);
    }
    else if (idx & 0x0c)
    {
        res = float4(gu * dvdX.y + dm * dvdX.x,
                     gu * dvdY.y + dm * dvdY.x,
                     gv * dvdX.y + dm * dvdX.z,
                     gv * dvdY.y + dm * dvdY.z);
    }
    else
    {
        res = float4(gu * dvdX.z + copysign(dm, c) * dvdX.x,
                     gu * dvdY.z + copysign(dm, c) * dvdY.x,
                     gv * dvdX.z - dm * dvdX.y,
                     gv * dvdY.z - dm * dvdY.y);
    }
    if (!isfinite(res.x) || !isfinite(res.y) || !isfinite(res.z) || !isfinite(res.w))
        return float4(0.f);
    return res;
}

//------------------------------------------------------------------------
// Texture indexing — nearest mode.

inline int indexTextureNearest(const constant TextureKernelParams& p, float3 uv, int tz, bool cubeMode)
{
    int w = p.texWidth;
    int h = p.texHeight;
    float u = uv.x;
    float v = uv.y;

    if (cubeMode)
    {
        int idx = indexCubeMap(u, v, uv.z);
        if (idx < 0) return -1;
        tz = 6 * tz + idx;
    }
    else
    {
        if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
        {
            u = u - floor(u);
            v = v - floor(v);
        }
    }

    u = u * float(w);
    v = v * float(h);

    int iu = int(floor(u));
    int iv = int(floor(v));

    if (!cubeMode && p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        if (iu < 0 || iu >= w || iv < 0 || iv >= h)
            return -1;
    }

    iu = clamp(iu, 0, w - 1);
    iv = clamp(iv, 0, h - 1);
    return iu + w * (iv + tz * h);
}

//------------------------------------------------------------------------
// Texture indexing — linear mode. Returns bilinear weights in float2.

inline float2 indexTextureLinear(const constant TextureKernelParams& p, float3 uv, int tz,
                                 thread int4& tcOut, int level, bool cubeMode)
{
    int2 sz = mipLevelSize(p, level);
    int w = sz.x;
    int h = sz.y;

    float u = uv.x;
    float v = uv.y;
    bool clampU = false;
    bool clampV = false;

    int face = 0;
    if (cubeMode)
    {
        face = indexCubeMap(u, v, uv.z);
        if (face < 0)
        {
            tcOut = int4(-1, -1, -1, -1);
            return float2(0.f, 0.f);
        }
        u = u * float(w) - 0.5f;
        v = v * float(h) - 0.5f;
    }
    else
    {
        if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
        {
            u = u - floor(u);
            v = v - floor(v);
        }

        u = u * float(w) - 0.5f;
        v = v * float(h) - 0.5f;

        if (p.boundaryMode == TEX_BOUNDARY_MODE_CLAMP)
        {
            u = clamp(u, 0.f, float(w) - 1.f);
            v = clamp(v, 0.f, float(h) - 1.f);
            clampU = (u == 0.f || u == float(w) - 1.f);
            clampV = (v == 0.f || v == float(h) - 1.f);
        }
    }

    int iu0 = int(floor(u));
    int iv0 = int(floor(v));
    int iu1 = iu0 + (clampU ? 0 : 1);
    int iv1 = iv0 + (clampV ? 0 : 1);
    u -= float(iu0);
    v -= float(iv0);

    // Cube map wrapping.
    bool cubeWrap = cubeMode && (iu0 < 0 || iv0 < 0 || iu1 >= w || iv1 >= h);
    if (cubeWrap)
    {
        tcOut = wrapCubeMap(face, iu0, iu1, iv0, iv1, w);
        tcOut += 6 * tz * w * h;
        return float2(u, v);
    }

    if (cubeMode)
        tz = 6 * tz + face;

    // Wrap overflowing texel indices.
    if (!cubeMode && p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        if (iu0 < 0) iu0 += w;
        if (iv0 < 0) iv0 += h;
        if (iu1 >= w) iu1 -= w;
        if (iv1 >= h) iv1 -= h;
    }

    int iu0z = iu0 + tz * w * h;
    int iu1z = iu1 + tz * w * h;
    tcOut.x = iu0z + w * iv0;
    tcOut.y = iu1z + w * iv0;
    tcOut.z = iu0z + w * iv1;
    tcOut.w = iu1z + w * iv1;

    // Zero boundary: invalidate addresses outside unit square.
    if (!cubeMode && p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        bool iu0_out = (iu0 < 0 || iu0 >= w);
        bool iu1_out = (iu1 < 0 || iu1 >= w);
        bool iv0_out = (iv0 < 0 || iv0 >= h);
        bool iv1_out = (iv1 < 0 || iv1 >= h);
        if (iu0_out || iv0_out) tcOut.x = -1;
        if (iu1_out || iv0_out) tcOut.y = -1;
        if (iu0_out || iv1_out) tcOut.z = -1;
        if (iu1_out || iv1_out) tcOut.w = -1;
    }

    return float2(u, v);
}

//------------------------------------------------------------------------
// Mip level calculation (forward only — no gradient outputs).

inline void calculateMipLevel(thread int& level0, thread int& level1, thread float& flevel,
                               const constant TextureKernelParams& p, int pidx, float3 uv,
                               bool cubeMode, bool biasOnly,
                               const device float* uvDABuf, const device float* mipBiasBuf)
{
    if (p.filterMode == TEX_MODE_NEAREST || p.filterMode == TEX_MODE_LINEAR)
        return;

    if (!biasOnly)
    {
        float4 uvDA;
        if (cubeMode)
        {
            float2 d0 = ((const device float2*)uvDABuf)[3 * pidx + 0];
            float2 d1 = ((const device float2*)uvDABuf)[3 * pidx + 1];
            float2 d2 = ((const device float2*)uvDABuf)[3 * pidx + 2];
            float3 dvdX = float3(d0.x, d1.x, d2.x);
            float3 dvdY = float3(d0.y, d1.y, d2.y);
            uvDA = indexCubeMapGradST(uv, dvdX, dvdY);
        }
        else
        {
            uvDA = ((const device float4*)uvDABuf)[pidx];
        }

        float uscl = float(p.texWidth);
        float vscl = float(p.texHeight);

        float dsdx = uvDA.x * uscl;
        float dsdy = uvDA.y * uscl;
        float dtdx = uvDA.z * vscl;
        float dtdy = uvDA.w * vscl;

        float A = dsdx * dsdx + dtdx * dtdx;
        float B = dsdy * dsdy + dtdy * dtdy;
        float C = dsdx * dsdy + dtdx * dtdy;
        float l2b = 0.5f * (A + B);
        float l2n = 0.25f * (A - B) * (A - B) + C * C;
        float l2a = sqrt(l2n);
        float lenMajorSqr = l2b + l2a;

        flevel = 0.5f * log2(lenMajorSqr);
    }

    // Bias.
    if (p.mipLevelBiasOfs >= 0)
        flevel += mipBiasBuf[pidx];

    flevel = clamp(flevel, 0.f, float(p.mipLevelMax));
    level0 = int(floor(flevel));

    if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR && flevel > 0.f)
    {
        level1 = min(level0 + 1, p.mipLevelMax);
        flevel -= float(level0);
    }
}

//------------------------------------------------------------------------
// Fetch quad helpers — 1-channel versions.

inline void fetchQuad1(thread float& a00, thread float& a10, thread float& a01, thread float& a11,
                       const device float* pIn, int4 tc, bool corner)
{
    if (corner)
    {
        float avg = 0.f;
        a00 = 0.f; a10 = 0.f; a01 = 0.f; a11 = 0.f;
        if (tc.x >= 0) avg += (a00 = pIn[tc.x]);
        if (tc.y >= 0) avg += (a10 = pIn[tc.y]);
        if (tc.z >= 0) avg += (a01 = pIn[tc.z]);
        if (tc.w >= 0) avg += (a11 = pIn[tc.w]);
        avg *= 0.33333333f;
        if (tc.x < 0) a00 = avg;
        if (tc.y < 0) a10 = avg;
        if (tc.z < 0) a01 = avg;
        if (tc.w < 0) a11 = avg;
    }
    else
    {
        a00 = (tc.x >= 0) ? pIn[tc.x] : 0.f;
        a10 = (tc.y >= 0) ? pIn[tc.y] : 0.f;
        a01 = (tc.z >= 0) ? pIn[tc.z] : 0.f;
        a11 = (tc.w >= 0) ? pIn[tc.w] : 0.f;
    }
}

// 2-channel fetch quad.
inline void fetchQuad2(thread float2& a00, thread float2& a10, thread float2& a01, thread float2& a11,
                       const device float* pIn, int4 tc, bool corner)
{
    if (corner)
    {
        float2 avg = float2(0.f);
        a00 = float2(0.f); a10 = float2(0.f); a01 = float2(0.f); a11 = float2(0.f);
        if (tc.x >= 0) avg += (a00 = *((const device float2*)&pIn[tc.x]));
        if (tc.y >= 0) avg += (a10 = *((const device float2*)&pIn[tc.y]));
        if (tc.z >= 0) avg += (a01 = *((const device float2*)&pIn[tc.z]));
        if (tc.w >= 0) avg += (a11 = *((const device float2*)&pIn[tc.w]));
        avg *= 0.33333333f;
        if (tc.x < 0) a00 = avg;
        if (tc.y < 0) a10 = avg;
        if (tc.z < 0) a01 = avg;
        if (tc.w < 0) a11 = avg;
    }
    else
    {
        a00 = (tc.x >= 0) ? *((const device float2*)&pIn[tc.x]) : float2(0.f);
        a10 = (tc.y >= 0) ? *((const device float2*)&pIn[tc.y]) : float2(0.f);
        a01 = (tc.z >= 0) ? *((const device float2*)&pIn[tc.z]) : float2(0.f);
        a11 = (tc.w >= 0) ? *((const device float2*)&pIn[tc.w]) : float2(0.f);
    }
}

// 4-channel fetch quad.
inline void fetchQuad4(thread float4& a00, thread float4& a10, thread float4& a01, thread float4& a11,
                       const device float* pIn, int4 tc, bool corner)
{
    if (corner)
    {
        float4 avg = float4(0.f);
        a00 = float4(0.f); a10 = float4(0.f); a01 = float4(0.f); a11 = float4(0.f);
        if (tc.x >= 0) avg += (a00 = *((const device float4*)&pIn[tc.x]));
        if (tc.y >= 0) avg += (a10 = *((const device float4*)&pIn[tc.y]));
        if (tc.z >= 0) avg += (a01 = *((const device float4*)&pIn[tc.z]));
        if (tc.w >= 0) avg += (a11 = *((const device float4*)&pIn[tc.w]));
        avg *= 0.33333333f;
        if (tc.x < 0) a00 = avg;
        if (tc.y < 0) a10 = avg;
        if (tc.z < 0) a01 = avg;
        if (tc.w < 0) a11 = avg;
    }
    else
    {
        a00 = (tc.x >= 0) ? *((const device float4*)&pIn[tc.x]) : float4(0.f);
        a10 = (tc.y >= 0) ? *((const device float4*)&pIn[tc.y]) : float4(0.f);
        a01 = (tc.z >= 0) ? *((const device float4*)&pIn[tc.z]) : float4(0.f);
        a11 = (tc.w >= 0) ? *((const device float4*)&pIn[tc.w]) : float4(0.f);
    }
}

//------------------------------------------------------------------------
// Mip build kernels — 2x2 box filter downsampling.

kernel void MipBuildKernel1(
    const device float* texIn   [[buffer(0)]],
    device float*       texOut  [[buffer(1)]],
    constant TextureKernelParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int2 sz_in  = mipLevelSize(p, p.mipLevelOut - 1);
    int2 sz_out = mipLevelSize(p, p.mipLevelOut);

    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    if (px >= sz_out.x || py >= sz_out.y)
        return;

    int pidx_in0 = p.channels * (((px + sz_in.x * py) << 1) + (pz * sz_in.x * sz_in.y));
    int pidx_in1 = pidx_in0 + p.channels * sz_in.x;
    int pidx_out = p.channels * (px + sz_out.x * (py + sz_out.y * pz));

    if (sz_in.x == 1 || sz_in.y == 1)
    {
        if (sz_in.y == 1)
            pidx_in1 = pidx_in0 + p.channels;
        for (int i = 0; i < p.channels; i += 1)
        {
            float v0 = texIn[pidx_in0 + i];
            float v1 = texIn[pidx_in1 + i];
            texOut[pidx_out + i] = 0.5f * (v0 + v1);
        }
        return;
    }

    for (int i = 0; i < p.channels; i += 1)
    {
        float v0 = texIn[pidx_in0 + i];
        float v1 = texIn[pidx_in0 + i + p.channels];
        float v2 = texIn[pidx_in1 + i];
        float v3 = texIn[pidx_in1 + i + p.channels];
        texOut[pidx_out + i] = 0.25f * (v0 + v1 + v2 + v3);
    }
}

kernel void MipBuildKernel2(
    const device float* texIn   [[buffer(0)]],
    device float*       texOut  [[buffer(1)]],
    constant TextureKernelParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int2 sz_in  = mipLevelSize(p, p.mipLevelOut - 1);
    int2 sz_out = mipLevelSize(p, p.mipLevelOut);

    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    if (px >= sz_out.x || py >= sz_out.y)
        return;

    int pidx_in0 = p.channels * (((px + sz_in.x * py) << 1) + (pz * sz_in.x * sz_in.y));
    int pidx_in1 = pidx_in0 + p.channels * sz_in.x;
    int pidx_out = p.channels * (px + sz_out.x * (py + sz_out.y * pz));

    if (sz_in.x == 1 || sz_in.y == 1)
    {
        if (sz_in.y == 1)
            pidx_in1 = pidx_in0 + p.channels;
        for (int i = 0; i < p.channels; i += 2)
        {
            float2 v0 = *((const device float2*)&texIn[pidx_in0 + i]);
            float2 v1 = *((const device float2*)&texIn[pidx_in1 + i]);
            *((device float2*)&texOut[pidx_out + i]) = 0.5f * (v0 + v1);
        }
        return;
    }

    for (int i = 0; i < p.channels; i += 2)
    {
        float2 v0 = *((const device float2*)&texIn[pidx_in0 + i]);
        float2 v1 = *((const device float2*)&texIn[pidx_in0 + i + p.channels]);
        float2 v2 = *((const device float2*)&texIn[pidx_in1 + i]);
        float2 v3 = *((const device float2*)&texIn[pidx_in1 + i + p.channels]);
        *((device float2*)&texOut[pidx_out + i]) = 0.25f * (v0 + v1 + v2 + v3);
    }
}

kernel void MipBuildKernel4(
    const device float* texIn   [[buffer(0)]],
    device float*       texOut  [[buffer(1)]],
    constant TextureKernelParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int2 sz_in  = mipLevelSize(p, p.mipLevelOut - 1);
    int2 sz_out = mipLevelSize(p, p.mipLevelOut);

    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    if (px >= sz_out.x || py >= sz_out.y)
        return;

    int pidx_in0 = p.channels * (((px + sz_in.x * py) << 1) + (pz * sz_in.x * sz_in.y));
    int pidx_in1 = pidx_in0 + p.channels * sz_in.x;
    int pidx_out = p.channels * (px + sz_out.x * (py + sz_out.y * pz));

    if (sz_in.x == 1 || sz_in.y == 1)
    {
        if (sz_in.y == 1)
            pidx_in1 = pidx_in0 + p.channels;
        for (int i = 0; i < p.channels; i += 4)
        {
            float4 v0 = *((const device float4*)&texIn[pidx_in0 + i]);
            float4 v1 = *((const device float4*)&texIn[pidx_in1 + i]);
            *((device float4*)&texOut[pidx_out + i]) = 0.5f * (v0 + v1);
        }
        return;
    }

    for (int i = 0; i < p.channels; i += 4)
    {
        float4 v0 = *((const device float4*)&texIn[pidx_in0 + i]);
        float4 v1 = *((const device float4*)&texIn[pidx_in0 + i + p.channels]);
        float4 v2 = *((const device float4*)&texIn[pidx_in1 + i]);
        float4 v3 = *((const device float4*)&texIn[pidx_in1 + i + p.channels]);
        *((device float4*)&texOut[pidx_out + i]) = 0.25f * (v0 + v1 + v2 + v3);
    }
}

//------------------------------------------------------------------------
// Forward texture kernel — 1 channel at a time (channel count = odd).
// Handles all filter modes and boundary modes via runtime branching.

kernel void TextureFwdKernel1(
    const device float* texBuf      [[buffer(0)]],
    const device float* uvBuf       [[buffer(1)]],
    const device float* uvDABuf     [[buffer(2)]],
    const device float* mipBiasBuf  [[buffer(3)]],
    device float*       outBuf      [[buffer(4)]],
    constant TextureKernelParams& p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);
    device float* pOut = outBuf + pidx * p.channels;
    bool cubeMode = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE);

    // Get UV.
    float3 uv;
    if (cubeMode)
        uv = ((const device float3*)uvBuf)[pidx];
    else
        uv = float3(((const device float2*)uvBuf)[pidx], 0.f);

    // Nearest mode.
    if (p.filterMode == TEX_MODE_NEAREST)
    {
        int tc = indexTextureNearest(p, uv, tz, cubeMode);
        tc *= p.channels;
        const device float* pIn = texBuf;
        for (int i = 0; i < p.channels; i += 1)
            pOut[i] = (tc >= 0) ? pIn[tc + i] : 0.f;
        return;
    }

    // Calculate mip level.
    float  flevel = 0.f;
    int    level0 = 0;
    int    level1 = 0;
    bool biasOnly = (p.enableMip != 0 && p.uvDAOfs < 0);
    calculateMipLevel(level0, level1, flevel, p, pidx, uv, cubeMode, biasOnly, uvDABuf, mipBiasBuf);

    // Get texel indices for level 0.
    int4 tc0 = int4(0);
    float2 uv0 = indexTextureLinear(p, uv, tz, tc0, level0, cubeMode);
    const device float* pIn0 = texBuf + p.texOfs[level0];
    bool corner0 = cubeMode && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Bilinear fetch (linear or linear-mipmap-nearest).
    if (p.filterMode == TEX_MODE_LINEAR || p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        for (int i = 0; i < p.channels; i += 1, tc0 += 1)
        {
            float a00, a10, a01, a11;
            fetchQuad1(a00, a10, a01, a11, pIn0, tc0, corner0);
            pOut[i] = bilerp_f(a00, a10, a01, a11, uv0);
        }
        return;
    }

    // Trilinear: get level 1.
    int4 tc1 = int4(0);
    float2 uv1 = indexTextureLinear(p, uv, tz, tc1, level1, cubeMode);
    const device float* pIn1 = texBuf + p.texOfs[level1];
    bool corner1 = cubeMode && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    for (int i = 0; i < p.channels; i += 1, tc0 += 1, tc1 += 1)
    {
        float a00, a10, a01, a11;
        fetchQuad1(a00, a10, a01, a11, pIn0, tc0, corner0);
        float a = bilerp_f(a00, a10, a01, a11, uv0);

        if (flevel > 0.f)
        {
            float b00, b10, b01, b11;
            fetchQuad1(b00, b10, b01, b11, pIn1, tc1, corner1);
            float b = bilerp_f(b00, b10, b01, b11, uv1);
            a = lerp_f(a, b, flevel);
        }

        pOut[i] = a;
    }
}

//------------------------------------------------------------------------
// Forward texture kernel — 2 channels at a time.

kernel void TextureFwdKernel2(
    const device float* texBuf      [[buffer(0)]],
    const device float* uvBuf       [[buffer(1)]],
    const device float* uvDABuf     [[buffer(2)]],
    const device float* mipBiasBuf  [[buffer(3)]],
    device float*       outBuf      [[buffer(4)]],
    constant TextureKernelParams& p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);
    device float* pOut = outBuf + pidx * p.channels;
    bool cubeMode = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE);

    float3 uv;
    if (cubeMode)
        uv = ((const device float3*)uvBuf)[pidx];
    else
        uv = float3(((const device float2*)uvBuf)[pidx], 0.f);

    if (p.filterMode == TEX_MODE_NEAREST)
    {
        int tc = indexTextureNearest(p, uv, tz, cubeMode);
        tc *= p.channels;
        const device float* pIn = texBuf;
        for (int i = 0; i < p.channels; i += 2)
            *((device float2*)&pOut[i]) = (tc >= 0) ? *((const device float2*)&pIn[tc + i]) : float2(0.f);
        return;
    }

    float  flevel = 0.f;
    int    level0 = 0;
    int    level1 = 0;
    bool biasOnly = (p.enableMip != 0 && p.uvDAOfs < 0);
    calculateMipLevel(level0, level1, flevel, p, pidx, uv, cubeMode, biasOnly, uvDABuf, mipBiasBuf);

    int4 tc0 = int4(0);
    float2 uv0 = indexTextureLinear(p, uv, tz, tc0, level0, cubeMode);
    const device float* pIn0 = texBuf + p.texOfs[level0];
    bool corner0 = cubeMode && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    if (p.filterMode == TEX_MODE_LINEAR || p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        for (int i = 0; i < p.channels; i += 2, tc0 += 2)
        {
            float2 a00, a10, a01, a11;
            fetchQuad2(a00, a10, a01, a11, pIn0, tc0, corner0);
            *((device float2*)&pOut[i]) = bilerp_f2(a00, a10, a01, a11, uv0);
        }
        return;
    }

    int4 tc1 = int4(0);
    float2 uv1 = indexTextureLinear(p, uv, tz, tc1, level1, cubeMode);
    const device float* pIn1 = texBuf + p.texOfs[level1];
    bool corner1 = cubeMode && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    for (int i = 0; i < p.channels; i += 2, tc0 += 2, tc1 += 2)
    {
        float2 a00, a10, a01, a11;
        fetchQuad2(a00, a10, a01, a11, pIn0, tc0, corner0);
        float2 a = bilerp_f2(a00, a10, a01, a11, uv0);

        if (flevel > 0.f)
        {
            float2 b00, b10, b01, b11;
            fetchQuad2(b00, b10, b01, b11, pIn1, tc1, corner1);
            float2 b = bilerp_f2(b00, b10, b01, b11, uv1);
            a = lerp_f2(a, b, flevel);
        }

        *((device float2*)&pOut[i]) = a;
    }
}

//------------------------------------------------------------------------
// Forward texture kernel — 4 channels at a time.

kernel void TextureFwdKernel4(
    const device float* texBuf      [[buffer(0)]],
    const device float* uvBuf       [[buffer(1)]],
    const device float* uvDABuf     [[buffer(2)]],
    const device float* mipBiasBuf  [[buffer(3)]],
    device float*       outBuf      [[buffer(4)]],
    constant TextureKernelParams& p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);
    device float* pOut = outBuf + pidx * p.channels;
    bool cubeMode = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE);

    float3 uv;
    if (cubeMode)
        uv = ((const device float3*)uvBuf)[pidx];
    else
        uv = float3(((const device float2*)uvBuf)[pidx], 0.f);

    if (p.filterMode == TEX_MODE_NEAREST)
    {
        int tc = indexTextureNearest(p, uv, tz, cubeMode);
        tc *= p.channels;
        const device float* pIn = texBuf;
        for (int i = 0; i < p.channels; i += 4)
            *((device float4*)&pOut[i]) = (tc >= 0) ? *((const device float4*)&pIn[tc + i]) : float4(0.f);
        return;
    }

    float  flevel = 0.f;
    int    level0 = 0;
    int    level1 = 0;
    bool biasOnly = (p.enableMip != 0 && p.uvDAOfs < 0);
    calculateMipLevel(level0, level1, flevel, p, pidx, uv, cubeMode, biasOnly, uvDABuf, mipBiasBuf);

    int4 tc0 = int4(0);
    float2 uv0 = indexTextureLinear(p, uv, tz, tc0, level0, cubeMode);
    const device float* pIn0 = texBuf + p.texOfs[level0];
    bool corner0 = cubeMode && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    if (p.filterMode == TEX_MODE_LINEAR || p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        for (int i = 0; i < p.channels; i += 4, tc0 += 4)
        {
            float4 a00, a10, a01, a11;
            fetchQuad4(a00, a10, a01, a11, pIn0, tc0, corner0);
            *((device float4*)&pOut[i]) = bilerp_f4(a00, a10, a01, a11, uv0);
        }
        return;
    }

    int4 tc1 = int4(0);
    float2 uv1 = indexTextureLinear(p, uv, tz, tc1, level1, cubeMode);
    const device float* pIn1 = texBuf + p.texOfs[level1];
    bool corner1 = cubeMode && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    for (int i = 0; i < p.channels; i += 4, tc0 += 4, tc1 += 4)
    {
        float4 a00, a10, a01, a11;
        fetchQuad4(a00, a10, a01, a11, pIn0, tc0, corner0);
        float4 a = bilerp_f4(a00, a10, a01, a11, uv0);

        if (flevel > 0.f)
        {
            float4 b00, b10, b01, b11;
            fetchQuad4(b00, b10, b01, b11, pIn1, tc1, corner1);
            float4 b = bilerp_f4(b00, b10, b01, b11, uv1);
            a = lerp_f4(a, b, flevel);
        }

        *((device float4*)&pOut[i]) = a;
    }
}

// ========== Backward Pass — Texture Gradient Kernels ==========

// Helper: atomically accumulate bilinear weights into 4 texels of grad texture.
inline void accumQuad1(float4 tw, device float* gradTex, int4 tc, bool corner)
{
    if (corner) {
        if (tc.x >= 0) tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.x], tw.x);
        if (tc.y >= 0) tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.y], tw.y);
        if (tc.z >= 0) tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.z], tw.z);
        if (tc.w >= 0) tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.w], tw.w);
    } else {
        tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.x], tw.x);
        tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.y], tw.y);
        tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.z], tw.z);
        tex_atomicAddFloat((device atomic_uint*)&gradTex[tc.w], tw.w);
    }
}

// Texture gradient kernel — handles all filter modes.
// Uses 1-channel-at-a-time loop (like TextureFwdKernel1) for simplicity.
// Covers nearest, linear, linear-mipmap-nearest, linear-mipmap-linear.
kernel void TextureGradKernel(
    const device float* texBuf      [[buffer(0)]],   // Combined tex+mip buffer
    const device float* uvBuf       [[buffer(1)]],   // [N, H, W, 2|3]
    const device float* uvDABuf     [[buffer(2)]],   // UV derivatives (may be dummy)
    const device float* mipBiasBuf  [[buffer(3)]],   // Mip level bias (may be dummy)
    const device float* dyBuf       [[buffer(4)]],   // [N, H, W, C] incoming grad
    device float*       gradTexBuf  [[buffer(5)]],   // gradTex (same layout as texBuf)
    device float*       gradUVBuf   [[buffer(6)]],   // [N, H, W, 2] UV gradient
    device float*       gradUVDABuf [[buffer(7)]],   // [N, H, W, 4] UV deriv gradient
    device float*       gradMipBuf  [[buffer(8)]],   // [N, H, W] mip bias gradient
    constant TextureKernelParams& p [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);
    bool cubeMode = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE);

    // Early exit: check if all output gradients are zero.
    const device float* pDy = dyBuf + pidx * p.channels;
    uint dmax = 0u;
    for (int i = 0; i < p.channels; i++)
        dmax |= as_type<uint>(pDy[i]);
    if (as_type<float>(dmax) == 0.f) {
        if (p.filterMode != TEX_MODE_NEAREST) {
            if (cubeMode)
                ((device float3*)gradUVBuf)[pidx] = float3(0.f);
            else
                ((device float2*)gradUVBuf)[pidx] = float2(0.f);
        }
        if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR) {
            if (gradUVDABuf) {
                if (cubeMode) {
                    ((device float2*)gradUVDABuf)[3 * pidx + 0] = float2(0.f);
                    ((device float2*)gradUVDABuf)[3 * pidx + 1] = float2(0.f);
                    ((device float2*)gradUVDABuf)[3 * pidx + 2] = float2(0.f);
                } else {
                    ((device float4*)gradUVDABuf)[pidx] = float4(0.f);
                }
            }
            if (gradMipBuf)
                gradMipBuf[pidx] = 0.f;
        }
        return;
    }

    // Get UV.
    float3 uv;
    if (cubeMode)
        uv = ((const device float3*)uvBuf)[pidx];
    else
        uv = float3(((const device float2*)uvBuf)[pidx], 0.f);

    // Nearest mode — texture gradients only.
    if (p.filterMode == TEX_MODE_NEAREST) {
        int tc = indexTextureNearest(p, uv, tz, cubeMode);
        if (tc < 0) return;
        tc *= p.channels;
        for (int i = 0; i < p.channels; i++)
            tex_atomicAddFloat((device atomic_uint*)&gradTexBuf[tc + i], pDy[i]);
        return;
    }

    // Calculate mip level.
    float4 dw = float4(0.f);
    float  flevel = 0.f;
    int    level0 = 0, level1 = 0;
    bool   biasOnly = (p.enableMip != 0 && p.uvDAOfs < 0);

    if (p.filterMode >= TEX_MODE_LINEAR_MIPMAP_NEAREST) {
        // Compute mip level (same as forward).
        calculateMipLevel(level0, level1, flevel, p, pidx, uv, cubeMode, biasOnly, uvDABuf, mipBiasBuf);
    }

    // UV gradient accumulators.
    float gu = 0.f, gv = 0.f;

    // Level 0 texel indices.
    int4 tc0 = int4(0);
    float2 uv0 = indexTextureLinear(p, uv, tz, tc0, level0, cubeMode);
    const device float* pIn0 = texBuf + p.texOfs[level0];
    device float* pGrad0 = gradTexBuf + p.texOfs[level0];
    bool corner0 = cubeMode && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Bilinear weights.
    float uv011 = uv0.x * uv0.y;
    float uv010 = uv0.x - uv011;
    float uv001 = uv0.y - uv011;
    float uv000 = 1.f - uv0.x - uv001;
    float4 tw0 = float4(uv000, uv010, uv001, uv011);

    int2 sz0 = mipLevelSize(p, level0);
    float sclu0 = float(sz0.x);
    float sclv0 = float(sz0.y);

    // Linear or linear-mipmap-nearest.
    if (p.filterMode == TEX_MODE_LINEAR || p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST) {
        for (int i = 0; i < p.channels; i++, tc0 += 1) {
            float dy = pDy[i];
            accumQuad1(tw0 * dy, pGrad0, tc0, corner0);

            float a00, a10, a01, a11;
            fetchQuad1(a00, a10, a01, a11, pIn0, tc0, corner0);
            float ad = (a11 + a00 - a10 - a01);
            gu += dy * ((a10 - a00) + uv0.y * ad) * sclu0;
            gv += dy * ((a01 - a00) + uv0.x * ad) * sclv0;
        }

        if (cubeMode)
            ;
        else
            ((device float2*)gradUVBuf)[pidx] = float2(gu, gv);
        return;
    }

    // Trilinear — linear-mipmap-linear.
    float df = 0.f; // dL/df (mip level gradient)

    int4 tc1 = int4(0);
    float2 uv1 = indexTextureLinear(p, uv, tz, tc1, level1, cubeMode);
    const device float* pIn1 = texBuf + p.texOfs[level1];
    device float* pGrad1 = gradTexBuf + p.texOfs[level1];
    bool corner1 = cubeMode && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    float uv111 = uv1.x * uv1.y;
    float uv110 = uv1.x - uv111;
    float uv101 = uv1.y - uv111;
    float uv100 = 1.f - uv1.x - uv101;
    float4 tw1 = float4(uv100, uv110, uv101, uv111);

    int2 sz1 = mipLevelSize(p, level1);
    float sclu1 = float(sz1.x);
    float sclv1 = float(sz1.y);

    for (int i = 0; i < p.channels; i++, tc0 += 1, tc1 += 1) {
        float dy = pDy[i];
        float dy0 = (1.f - flevel) * dy;
        accumQuad1(tw0 * dy0, pGrad0, tc0, corner0);

        float a00, a10, a01, a11;
        fetchQuad1(a00, a10, a01, a11, pIn0, tc0, corner0);
        float ad = (a11 + a00 - a10 - a01);
        gu += dy0 * ((a10 - a00) + uv0.y * ad) * sclu0;
        gv += dy0 * ((a01 - a00) + uv0.x * ad) * sclv0;

        if (flevel > 0.f) {
            float dy1 = flevel * dy;
            accumQuad1(tw1 * dy1, pGrad1, tc1, corner1);

            float b00, b10, b01, b11;
            fetchQuad1(b00, b10, b01, b11, pIn1, tc1, corner1);
            float bd = (b11 + b00 - b10 - b01);
            gu += dy1 * ((b10 - b00) + uv1.y * bd) * sclu1;
            gv += dy1 * ((b01 - b00) + uv1.x * bd) * sclv1;

            float a = bilerp_f(a00, a10, a01, a11, uv0);
            float b = bilerp_f(b00, b10, b01, b11, uv1);
            df += (b - a) * dy;
        }
    }

    // Store UV gradients.
    if (cubeMode)
        ;
    else
        ((device float2*)gradUVBuf)[pidx] = float2(gu, gv);

    // Mip level bias gradient.
    if (gradMipBuf)
        gradMipBuf[pidx] = df;

    // UV pixel differential gradients.
    if (!biasOnly && gradUVDABuf) {
        // For non-cube: dw * df gives UV-DA gradients.
        // Recompute dw from forward mip level calculation.
        if (!cubeMode) {
            float4 uvDA = ((const device float4*)uvDABuf)[pidx];
            float uscl = float(p.texWidth);
            float vscl = float(p.texHeight);
            float dsdx = uvDA.x * uscl;
            float dsdy = uvDA.y * uscl;
            float dtdx = uvDA.z * vscl;
            float dtdy = uvDA.w * vscl;
            float A = dsdx * dsdx + dtdx * dtdx;
            float B = dsdy * dsdy + dtdy * dtdy;
            float C = dsdx * dsdy + dtdx * dtdy;
            float l2b = 0.5f * (A + B);
            float l2n = 0.25f * (A - B) * (A - B) + C * C;
            float l2a = sqrt(l2n);
            float lenMajorSqr = l2b + l2a;
            // df/d(lenMajorSqr) = 0.5 / (lenMajorSqr * ln(2))
            float dfl = (lenMajorSqr > 0.f) ? 0.5f / (lenMajorSqr * 0.6931471805599453f) : 0.f;
            // d(lenMajorSqr)/d(A,B,C)
            float inv_l2a = (l2a > 0.f) ? 1.f / l2a : 0.f;
            float dA = 0.5f + 0.5f * (A - B) * inv_l2a;
            float dB = 0.5f - 0.5f * (A - B) * inv_l2a;
            float dC = 2.f * C * inv_l2a;
            // Chain rule to uvDA components.
            float g_dsdx = dfl * df * (dA * 2.f * dsdx * uscl + dC * dsdy * uscl);
            float g_dsdy = dfl * df * (dB * 2.f * dsdy * uscl + dC * dsdx * uscl);
            float g_dtdx = dfl * df * (dA * 2.f * dtdx * vscl + dC * dtdy * vscl);
            float g_dtdy = dfl * df * (dB * 2.f * dtdy * vscl + dC * dtdx * vscl);
            ((device float4*)gradUVDABuf)[pidx] = float4(g_dsdx, g_dsdy, g_dtdx, g_dtdy);
        }
    }
}

// Mip gradient kernel — pulls gradients from all mip levels back to level 0.
kernel void MipGradKernel(
    device float*       gradTexBuf   [[buffer(0)]],  // Combined grad tex buffer
    constant TextureKernelParams& p  [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    if (px >= p.texWidth || py >= p.texHeight)
        return;

    int c = p.channels;

    // Accumulate from all mip levels into base level.
    int x = px;
    int y = py;
    float w = 1.f;

    int2 sz = int2(p.texWidth, p.texHeight);
    for (int level = 1; level <= p.mipLevelMax; level++) {
        if (sz.x > 1) w *= 0.5f;
        if (sz.y > 1) w *= 0.5f;

        sz = mipLevelSize(p, level);
        x >>= 1;
        y >>= 1;

        device float* pIn = gradTexBuf + p.texOfs[level] + (x + sz.x * (y + sz.y * pz)) * c;
        for (int i = 0; i < c; i++) {
            float val = pIn[i] * w;
            if (val != 0.f)
                tex_atomicAddFloat((device atomic_uint*)&gradTexBuf[p.texOfs[0] + (px + p.texWidth * (py + p.texHeight * pz)) * c + i], val);
        }
    }
}

//------------------------------------------------------------------------
