// Metal differentiable antialiasing kernels.
// Based on Laine et al. 2020, Section 4 — antialiased silhouette edges.
#include <metal_stdlib>
using namespace metal;

//------------------------------------------------------------------------
// Constants.

#define AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH         32
#define AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT        8
#define AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK        256
#define AA_MESH_KERNEL_THREADS_PER_BLOCK            256
#define AA_GRAD_KERNEL_THREADS_PER_BLOCK            256

// With more than 16777216 triangles (alloc >= 33554432) use 4, otherwise 8.
inline int aa_hash_elements_per_triangle(int alloc) { return (alloc >= (2 << 25)) ? 4 : 8; }
inline int aa_log_hash_elements_per_triangle(int alloc) { return (alloc >= (2 << 25)) ? 2 : 3; }

#define F32_MAX (3.402823466e+38f)
#define JENKINS_MAGIC (0x9e3779b9u)

// Constant-buffer params for kernel dispatch (matches host-side AntialiasKernelParams layout
// for scalar fields; device pointers are passed as separate buffer bindings).
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
// Atomic float add.

#if __METAL_VERSION__ >= 310
// Metal 3.1 (M3+): native float atomic add
inline void atomicAddFloat(device atomic_float* addr, float value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
#else
// Metal 3.0 fallback: CAS-loop emulation
inline void atomicAddFloat(device atomic_uint* addr, float value) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + value;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired_bits,
                                                   memory_order_relaxed, memory_order_relaxed))
            break;
    }
}
#endif

//------------------------------------------------------------------------
// AAWorkItem flags — matches host-side struct layout.

constant int AA_EDGE_MASK       = 3;
constant int AA_FLAG_DOWN_BIT   = 2;
constant int AA_FLAG_TRI1_BIT   = 3;

//------------------------------------------------------------------------
// Triangle ID <-> float32 conversion (must match rasterize.metal).

inline int float_to_triidx(float x) {
    if (x <= 16777216.0f) return int(x);
    return as_type<int>(x) - 0x4a800000;
}

//------------------------------------------------------------------------
// Helpers.

inline bool same_sign(float a, float b) {
    return (as_type<int>(a) ^ as_type<int>(b)) >= 0;
}

inline bool rational_gt(float n0, float n1, float d0, float d1) {
    return (n0 * d1 > n1 * d0) == same_sign(d0, d1);
}

inline int max_idx3(float n0, float n1, float n2, float d0, float d1, float d2)
{
    bool g10 = rational_gt(n1, n0, d1, d0);
    bool g20 = rational_gt(n2, n0, d2, d0);
    bool g21 = rational_gt(n2, n1, d2, d1);
    if (g20 && g21) return 2;
    if (g10) return 1;
    return 0;
}

//------------------------------------------------------------------------
// Jenkins hash mixing.

inline void jenkins_mix(thread uint& a, thread uint& b, thread uint& c)
{
    a -= b; a -= c; a ^= (c >> 13);
    b -= c; b -= a; b ^= (a << 8);
    c -= a; c -= b; c ^= (b >> 13);
    a -= b; a -= c; a ^= (c >> 12);
    b -= c; b -= a; b ^= (a << 16);
    c -= a; c -= b; c ^= (b >> 5);
    a -= b; a -= c; a ^= (c >> 3);
    b -= c; b -= a; b ^= (a << 10);
    c -= a; c -= b; c ^= (b >> 15);
}

//------------------------------------------------------------------------
// Hash index class — open-addressing with odd-skip linear probing.

struct HashIndex
{
    uint m_idx;
    uint m_skip;
    uint m_mask;

    HashIndex(int allocTriangles, ulong key)
    {
        m_mask = (uint(allocTriangles) << uint(aa_log_hash_elements_per_triangle(allocTriangles))) - 1u;
        m_idx  = uint(key & 0xffffffffu);
        m_skip = uint(key >> 32);
        uint dummy = JENKINS_MAGIC;
        jenkins_mix(m_idx, m_skip, dummy);
        m_idx &= m_mask;
        m_skip &= m_mask;
        m_skip |= 1u;
    }

    int get() const { return int(m_idx); }
    void next() { m_idx = (m_idx + m_skip) & m_mask; }
};

//------------------------------------------------------------------------
// Hash insert — uses atomic CAS on 64-bit key, then on 32-bit values.
// evHash is treated as an array of 4 x uint per entry (like uint4).
// Entry layout: [key_lo, key_hi, val0, val1]

inline void hash_insert(int allocTriangles, device atomic_uint* evHash, ulong key, int v)
{
    HashIndex idx(allocTriangles, key);

    uint key_lo = uint(key & 0xffffffffu);
    uint key_hi = uint(key >> 32);

    while (true)
    {
        int base = idx.get() * 4; // 4 uints per entry

        // Try to claim the slot by writing key_lo. If slot is 0 (empty), we win.
        uint prev_lo = 0u;
        bool ok_lo = atomic_compare_exchange_weak_explicit(&evHash[base + 0], &prev_lo, key_lo,
                                                           memory_order_relaxed, memory_order_relaxed);

        if (ok_lo)
        {
            // We claimed it — now write key_hi unconditionally.
            atomic_store_explicit(&evHash[base + 1], key_hi, memory_order_relaxed);
        }
        else if (prev_lo != key_lo)
        {
            // Collision with a different key — probe next.
            idx.next();
            continue;
        }
        else
        {
            // Same key_lo — verify key_hi matches. Spin until key_hi is written.
            uint stored_hi = atomic_load_explicit(&evHash[base + 1], memory_order_relaxed);
            if (stored_hi != 0u && stored_hi != key_hi)
            {
                // Hash collision on lower 32 bits but different full key. Probe next.
                idx.next();
                continue;
            }
            // If stored_hi is 0, it may still be getting written. For correctness on
            // the mesh kernel (single-pass insert), treat key_lo match as sufficient
            // since the probability of collision is negligible with Jenkins mixing.
        }

        // Insert value into val0 or val1.
        uint uv = uint(v);
        uint a = 0u;
        bool cas_ok = atomic_compare_exchange_weak_explicit(&evHash[base + 2], &a, uv,
                                                            memory_order_relaxed, memory_order_relaxed);
        if (!cas_ok && a != uv)
        {
            uint b = 0u;
            atomic_compare_exchange_weak_explicit(&evHash[base + 3], &b, uv,
                                                  memory_order_relaxed, memory_order_relaxed);
        }
        break;
    }
}

//------------------------------------------------------------------------
// Hash find — returns int2(val0, val1) for the given key.

inline int2 hash_find(int allocTriangles, device atomic_uint* evHash, ulong key)
{
    HashIndex idx(allocTriangles, key);
    uint key_lo = uint(key & 0xffffffffu);
    uint key_hi = uint(key >> 32);

    while (true)
    {
        int base = idx.get() * 4;
        uint k0 = atomic_load_explicit(&evHash[base + 0], memory_order_relaxed);
        uint k1 = atomic_load_explicit(&evHash[base + 1], memory_order_relaxed);
        ulong k = ulong(k0) | (ulong(k1) << 32);

        if (k == key || k == ulong(0))
        {
            uint v0 = atomic_load_explicit(&evHash[base + 2], memory_order_relaxed);
            uint v1 = atomic_load_explicit(&evHash[base + 3], memory_order_relaxed);
            return int2(int(v0), int(v1));
        }
        idx.next();
    }
}

// Non-atomic version for read-only access (forward analysis / grad).
inline int2 hash_find_ro(int allocTriangles, device const uint* evHash, ulong key)
{
    HashIndex idx(allocTriangles, key);

    while (true)
    {
        int base = idx.get() * 4;
        uint k0 = evHash[base + 0];
        uint k1 = evHash[base + 1];
        ulong k = ulong(k0) | (ulong(k1) << 32);

        if (k == key || k == ulong(0))
            return int2(int(evHash[base + 2]), int(evHash[base + 3]));
        idx.next();
    }
}

//------------------------------------------------------------------------
// Edge-vertex hash insert/find wrappers.

inline void evhash_insert_vertex(int allocTriangles, device atomic_uint* evHash, int va, int vb, int vn)
{
    if (va == vb)
        return;

    ulong v0 = ulong(uint(min(va, vb)) + 1u);
    ulong v1 = ulong(uint(max(va, vb)) + 1u);
    ulong vk = v0 | (v1 << 32);
    hash_insert(allocTriangles, evHash, vk, vn + 1);
}

inline int evhash_find_vertex(int allocTriangles, device const uint* evHash, int va, int vb, int vr)
{
    if (va == vb)
        return -1;

    ulong v0 = ulong(uint(min(va, vb)) + 1u);
    ulong v1 = ulong(uint(max(va, vb)) + 1u);
    ulong vk = v0 | (v1 << 32);
    int2 vn = hash_find_ro(allocTriangles, evHash, vk) - 1;
    if (vn.x == vr) return vn.y;
    if (vn.y == vr) return vn.x;
    return -1;
}

//------------------------------------------------------------------------
// Kernel 1: AntialiasFwdMeshKernel — builds edge-vertex hash table.

kernel void AntialiasFwdMeshKernel(
    device const int*       tri         [[buffer(0)]],
    device atomic_uint*     evHash      [[buffer(1)]],
    constant AntialiasConstParams& p    [[buffer(2)]],
    uint                    tid         [[thread_position_in_grid]]
)
{
    int idx = int(tid);
    if (idx >= p.numTriangles)
        return;

    int v0 = tri[idx * 3 + 0];
    int v1 = tri[idx * 3 + 1];
    int v2 = tri[idx * 3 + 2];

    if (v0 < 0 || v0 >= p.numVertices ||
        v1 < 0 || v1 >= p.numVertices ||
        v2 < 0 || v2 >= p.numVertices)
        return;

    if (v0 == v1 || v1 == v2 || v2 == v0)
        return;

    evhash_insert_vertex(p.allocTriangles, evHash, v1, v2, v0);
    evhash_insert_vertex(p.allocTriangles, evHash, v2, v0, v1);
    evhash_insert_vertex(p.allocTriangles, evHash, v0, v1, v2);
}

//------------------------------------------------------------------------
// Kernel 2: AntialiasFwdDiscontinuityKernel — marks pixels at silhouette edges.
// Uses threadgroup atomics for coalesced work counter update.

kernel void AntialiasFwdDiscontinuityKernel(
    device const float*     rasterOut   [[buffer(0)]],
    device atomic_int*      workCounter [[buffer(1)]],  // workBuffer[0] as atomic counter
    device int*             workItems   [[buffer(2)]],  // workBuffer raw int array for item writes
    constant AntialiasConstParams& p    [[buffer(3)]],
    uint3                   gid         [[thread_position_in_grid]]
)
{
    int px = int(gid.x);
    int py = int(gid.y);
    int pz = int(gid.z);
    if (px >= p.width || py >= p.height || pz >= p.n)
        return;

    // Pointer to our TriIdx and fetch.
    int pidx0 = ((px + p.width * (py + p.height * pz)) << 2) + 3;
    float tri0 = rasterOut[pidx0];

    // Look right, clamp at edge.
    int pidx1 = pidx0;
    if (px < p.width - 1)
        pidx1 += 4;
    float tri1 = rasterOut[pidx1];

    // Look down, clamp at edge.
    int pidx2 = pidx0;
    if (py < p.height - 1)
        pidx2 += p.width << 2;
    float tri2 = rasterOut[pidx2];

    // Determine amount of work.
    int count = 0;
    if (tri1 != tri0) count  = 1;
    if (tri2 != tri0) count += 1;
    if (!count)
        return;

    // workCounter[0] is the work count (first int of workBuffer).
    int idx = atomic_fetch_add_explicit(&workCounter[0], count, memory_order_relaxed);
    idx += 1; // Skip first slot (4 ints reserved for counters).

    // Write to memory. Each work item is 4 ints (int4).
    if (tri1 != tri0)
    {
        int base = idx * 4;
        workItems[base + 0] = px;
        workItems[base + 1] = py;
        workItems[base + 2] = (pz << 16);
        workItems[base + 3] = 0;
        idx++;
    }
    if (tri2 != tri0)
    {
        int base = idx * 4;
        workItems[base + 0] = px;
        workItems[base + 1] = py;
        workItems[base + 2] = (pz << 16) + (1 << AA_FLAG_DOWN_BIT);
        workItems[base + 3] = 0;
    }
}

//------------------------------------------------------------------------
// Kernel 3: AntialiasFwdAnalysisKernel — computes antialiased pixel colors.
// This is a 1D dispatch; each thread processes one work item.

kernel void AntialiasFwdAnalysisKernel(
    device const float*     color       [[buffer(0)]],
    device const float*     rasterOut   [[buffer(1)]],
    device const int*       tri         [[buffer(2)]],
    device const float*     pos         [[buffer(3)]],
    device float*           output      [[buffer(4)]],
    device int*             workBuffer  [[buffer(5)]],  // Raw int array (int4-aligned work items)
    device const uint*      evHash      [[buffer(6)]],
    constant AntialiasConstParams& p    [[buffer(7)]],
    uint                    tid         [[thread_position_in_grid]]
)
{
    // Read work count from first slot (workBuffer[0]).
    int workCount = workBuffer[0];
    int thread_idx = int(tid);
    if (thread_idx >= workCount)
        return;

    // Each work item is 4 ints. Slot 0 (4 ints) is reserved for counters.
    int itemBase = (thread_idx + 1) * 4;
    int px = workBuffer[itemBase + 0];
    int py = workBuffer[itemBase + 1];
    int pzFlags = workBuffer[itemBase + 2];
    int pz = int(uint(pzFlags) >> 16);
    int d  = (pzFlags >> AA_FLAG_DOWN_BIT) & 1;

    int pixel0 = px + p.width * (py + p.height * pz);
    int pixel1 = pixel0 + (d ? p.width : 1);

    // Read rasterizer output: float2 at (pixel << 1) + 1 gives (z, triIdx).
    // rasterOut is laid out as 4 floats per pixel: [u, v, z, triIdx].
    float z0  = rasterOut[pixel0 * 4 + 2];
    float t0f = rasterOut[pixel0 * 4 + 3];
    float z1  = rasterOut[pixel1 * 4 + 2];
    float t1f = rasterOut[pixel1 * 4 + 3];
    int tri0 = float_to_triidx(t0f) - 1;
    int tri1 = float_to_triidx(t1f) - 1;

    // Select triangle based on background / depth.
    int triSel = (tri0 >= 0) ? tri0 : tri1;
    if (tri0 >= 0 && tri1 >= 0)
        triSel = (z0 < z1) ? tri0 : tri1;
    if (triSel == tri1)
    {
        px += 1 - d;
        py += d;
    }

    // Bail out if triangle index is corrupt.
    if (triSel < 0 || triSel >= p.numTriangles)
        return;

    // Fetch vertex indices.
    int vi0 = tri[triSel * 3 + 0];
    int vi1 = tri[triSel * 3 + 1];
    int vi2 = tri[triSel * 3 + 2];

    // Bail out if vertex indices are corrupt.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // Fetch opposite vertex indices.
    int op0 = evhash_find_vertex(p.allocTriangles, evHash, vi2, vi1, vi0);
    int op1 = evhash_find_vertex(p.allocTriangles, evHash, vi0, vi2, vi1);
    int op2 = evhash_find_vertex(p.allocTriangles, evHash, vi1, vi0, vi2);

    // Instance mode: adjust vertex indices.
    if (p.instance_mode)
    {
        int vbase = pz * p.numVertices;
        vi0 += vbase;
        vi1 += vbase;
        vi2 += vbase;
        if (op0 >= 0) op0 += vbase;
        if (op1 >= 0) op1 += vbase;
        if (op2 >= 0) op2 += vbase;
    }

    // Fetch vertex positions (float4 each).
    float4 p0 = ((device const float4*)pos)[vi0];
    float4 p1 = ((device const float4*)pos)[vi1];
    float4 p2 = ((device const float4*)pos)[vi2];
    float4 o0 = (op0 < 0) ? p0 : ((device const float4*)pos)[op0];
    float4 o1 = (op1 < 0) ? p1 : ((device const float4*)pos)[op1];
    float4 o2 = (op2 < 0) ? p2 : ((device const float4*)pos)[op2];

    // Project vertices to pixel space.
    float w0  = 1.f / p0.w;
    float w1  = 1.f / p1.w;
    float w2  = 1.f / p2.w;
    float ow0 = 1.f / o0.w;
    float ow1 = 1.f / o1.w;
    float ow2 = 1.f / o2.w;
    float fx  = float(px) + .5f - p.xh;
    float fy  = float(py) + .5f - p.yh;
    float x0  = p0.x * w0 * p.xh - fx;
    float y0  = p0.y * w0 * p.yh - fy;
    float x1  = p1.x * w1 * p.xh - fx;
    float y1  = p1.y * w1 * p.yh - fy;
    float x2  = p2.x * w2 * p.xh - fx;
    float y2  = p2.y * w2 * p.yh - fy;
    float ox0 = o0.x * ow0 * p.xh - fx;
    float oy0 = o0.y * ow0 * p.yh - fy;
    float ox1 = o1.x * ow1 * p.xh - fx;
    float oy1 = o1.y * ow1 * p.yh - fy;
    float ox2 = o2.x * ow2 * p.xh - fx;
    float oy2 = o2.y * ow2 * p.yh - fy;

    // Signs to kill non-silhouette edges.
    float bb = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
    float a0 = (x1-ox0)*(y2-oy0) - (x2-ox0)*(y1-oy0);
    float a1 = (x2-ox1)*(y0-oy1) - (x0-ox1)*(y2-oy1);
    float a2 = (x0-ox2)*(y1-oy2) - (x1-ox2)*(y0-oy2);

    // If no matching signs anywhere, skip the rest.
    if (same_sign(a0, bb) || same_sign(a1, bb) || same_sign(a2, bb))
    {
        // XY flip for horizontal edges.
        if (d)
        {
            float tmp;
            tmp = x0; x0 = y0; y0 = tmp;
            tmp = x1; x1 = y1; y1 = tmp;
            tmp = x2; x2 = y2; y2 = tmp;
        }

        float dx0 = x2 - x1;
        float dx1 = x0 - x2;
        float dx2 = x1 - x0;
        float dy0 = y2 - y1;
        float dy1 = y0 - y2;
        float dy2 = y1 - y0;

        // Check if an edge crosses between us and the neighbor pixel.
        float dc = -F32_MAX;
        float ds = (triSel == tri0) ? 1.f : -1.f;
        float d0 = ds * (x1*dy0 - y1*dx0);
        float d1 = ds * (x2*dy1 - y2*dx1);
        float d2 = ds * (x0*dy2 - y0*dx2);

        if (same_sign(y1, y2)) { d0 = -F32_MAX; dy0 = 1.f; }
        if (same_sign(y2, y0)) { d1 = -F32_MAX; dy1 = 1.f; }
        if (same_sign(y0, y1)) { d2 = -F32_MAX; dy2 = 1.f; }

        int di = max_idx3(d0, d1, d2, dy0, dy1, dy2);
        if (di == 0 && same_sign(a0, bb) && fabs(dy0) >= fabs(dx0)) dc = d0 / dy0;
        if (di == 1 && same_sign(a1, bb) && fabs(dy1) >= fabs(dx1)) dc = d1 / dy1;
        if (di == 2 && same_sign(a2, bb) && fabs(dy2) >= fabs(dx2)) dc = d2 / dy2;
        float eps = .0625f;

        // Adjust output image if a suitable edge was found.
        if (dc > -eps && dc < 1.f + eps)
        {
            dc = min(max(dc, 0.f), 1.f);
            float alpha = ds * (.5f - dc);
            device const float* pColor0 = color + pixel0 * p.channels;
            device const float* pColor1 = color + pixel1 * p.channels;
            device float* pOutput = output + (alpha > 0.f ? pixel0 : pixel1) * p.channels;

            // Atomic add to output.
#if __METAL_VERSION__ >= 310
            device atomic_float* pOutAtomic = (device atomic_float*)pOutput;
#else
            device atomic_uint* pOutAtomic = (device atomic_uint*)pOutput;
#endif
            for (int i = 0; i < p.channels; i++)
                atomicAddFloat(&pOutAtomic[i], alpha * (pColor1[i] - pColor0[i]));

            // Rewrite the work item's flags and alpha.
            uint flags = uint(pz) << 16;
            flags |= uint(di);
            flags |= uint(d) << AA_FLAG_DOWN_BIT;
            flags |= (as_type<uint>(ds) >> 31) << AA_FLAG_TRI1_BIT;
            // Write z component (flags) and w component (alpha as int bits).
            workBuffer[itemBase + 2] = int(flags);
            workBuffer[itemBase + 3] = as_type<int>(alpha);
        }
    }
}

//------------------------------------------------------------------------
// Kernel 4: AntialiasGradKernel — backward pass.
// 1D dispatch, one thread per work item.

kernel void AntialiasGradKernel(
    device const float*     color       [[buffer(0)]],
    device const float*     rasterOut   [[buffer(1)]],
    device const int*       tri         [[buffer(2)]],
    device const float*     pos         [[buffer(3)]],
    device const float*     dy          [[buffer(4)]],
    device float*           gradColor   [[buffer(5)]],
    device float*           gradPos     [[buffer(6)]],
    device const int*       workBuffer  [[buffer(7)]],  // Raw int array
    constant AntialiasConstParams& p    [[buffer(8)]],
    uint                    tid         [[thread_position_in_grid]]
)
{
    int workCount = workBuffer[0];
    int thread_idx = int(tid);
    if (thread_idx >= workCount)
        return;

    // Read work item filled out by forward kernel.
    int itemBase = (thread_idx + 1) * 4;
    int item_w = workBuffer[itemBase + 3];
    if (item_w == 0)
        return; // No effect.

    // Unpack work item.
    int px = workBuffer[itemBase + 0];
    int py = workBuffer[itemBase + 1];
    int item_z = workBuffer[itemBase + 2];
    int pz = int(uint(item_z) >> 16);
    int d = (item_z >> AA_FLAG_DOWN_BIT) & 1;
    float alpha = as_type<float>(item_w);
    int tri1_flag = (item_z >> AA_FLAG_TRI1_BIT) & 1;
    int di = item_z & AA_EDGE_MASK;
    float ds = as_type<float>(as_type<int>(1.0f) | (tri1_flag << 31));
    int pixel0 = px + p.width * (py + p.height * pz);
    int pixel1 = pixel0 + (d ? p.width : 1);
    int triIdx = float_to_triidx(rasterOut[((tri1_flag ? pixel1 : pixel0) << 2) + 3]) - 1;
    if (tri1_flag)
    {
        px += 1 - d;
        py += d;
    }

    // Bail out if triangle index is corrupt.
    if (triIdx < 0 || triIdx >= p.numTriangles)
        return;

    // Outgoing color gradients.
#if __METAL_VERSION__ >= 310
    device atomic_float* pGrad0 = (device atomic_float*)(gradColor + pixel0 * p.channels);
    device atomic_float* pGrad1 = (device atomic_float*)(gradColor + pixel1 * p.channels);
#else
    device atomic_uint* pGrad0 = (device atomic_uint*)(gradColor + pixel0 * p.channels);
    device atomic_uint* pGrad1 = (device atomic_uint*)(gradColor + pixel1 * p.channels);
#endif

    // Incoming color gradients.
    device const float* pDy = dy + (alpha > 0.f ? pixel0 : pixel1) * p.channels;

    // Position gradient weight based on colors and incoming gradients.
    float dd = 0.f;
    device const float* pColor0 = color + pixel0 * p.channels;
    device const float* pColor1 = color + pixel1 * p.channels;

    for (int i = 0; i < p.channels; i++)
    {
        float dyv = pDy[i];
        if (dyv != 0.f)
        {
            dd += dyv * (pColor1[i] - pColor0[i]);
            float v = alpha * dyv;
            atomicAddFloat(&pGrad0[i], -v);
            atomicAddFloat(&pGrad1[i],  v);
        }
    }

    // If position weight is zero, skip the rest.
    if (dd == 0.f)
        return;

    // Fetch vertex indices of the active edge.
    int i1 = (di < 2) ? (di + 1) : 0;
    int i2 = (i1 < 2) ? (i1 + 1) : 0;
    int vi1 = tri[3 * triIdx + i1];
    int vi2 = tri[3 * triIdx + i2];

    // Bail out if vertex indices are corrupt.
    if (vi1 < 0 || vi1 >= p.numVertices || vi2 < 0 || vi2 >= p.numVertices)
        return;

    // Instance mode.
    if (p.instance_mode)
    {
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Fetch vertex positions.
    float4 vp1 = ((device const float4*)pos)[vi1];
    float4 vp2 = ((device const float4*)pos)[vi2];

    // Project vertices to pixel space.
    float pxh = p.xh;
    float pyh = p.yh;
    float fx = float(px) + .5f - pxh;
    float fy = float(py) + .5f - pyh;

    // XY flip for horizontal edges.
    if (d)
    {
        float tmp;
        tmp = vp1.x; vp1.x = vp1.y; vp1.y = tmp;
        tmp = vp2.x; vp2.x = vp2.y; vp2.y = tmp;
        tmp = pxh; pxh = pyh; pyh = tmp;
        tmp = fx; fx = fy; fy = tmp;
    }

    // Gradient calculation setup.
    float gw1 = 1.f / vp1.w;
    float gw2 = 1.f / vp2.w;
    float gx1 = vp1.x * gw1 * pxh - fx;
    float gy1 = vp1.y * gw1 * pyh - fy;
    float gx2 = vp2.x * gw2 * pxh - fx;
    float gy2 = vp2.y * gw2 * pyh - fy;
    float gdx = gx2 - gx1;
    float gdy = gy2 - gy1;
    float gdb = gx1 * gdy - gy1 * gdx;

    // Compute inverse delta-y with epsilon.
    float ep = copysign(1e-3f, gdy);
    float iy = 1.f / (gdy + ep);

    // Compute position gradients.
    float dby = gdb * iy;
    float iw1 = -gw1 * iy * dd;
    float iw2 =  gw2 * iy * dd;
    float gp1x = iw1 * pxh * gy2;
    float gp2x = iw2 * pxh * gy1;
    float gp1y = iw1 * pyh * (dby - gx2);
    float gp2y = iw2 * pyh * (dby - gx1);
    float gp1w = -(vp1.x * gp1x + vp1.y * gp1y) * gw1;
    float gp2w = -(vp2.x * gp2x + vp2.y * gp2y) * gw2;

    // XY flip the gradients.
    if (d)
    {
        float tmp;
        tmp = gp1x; gp1x = gp1y; gp1y = tmp;
        tmp = gp2x; gp2x = gp2y; gp2y = tmp;
    }

    // Kill position gradients if alpha was saturated.
    if (fabs(alpha) >= 0.5f)
    {
        gp1x = gp1y = gp1w = 0.f;
        gp2x = gp2y = gp2w = 0.f;
    }

    // Accumulate position gradients.
#if __METAL_VERSION__ >= 310
    device atomic_float* gp1_a = (device atomic_float*)(gradPos + 4 * vi1);
    device atomic_float* gp2_a = (device atomic_float*)(gradPos + 4 * vi2);
#else
    device atomic_uint* gp1_a = (device atomic_uint*)(gradPos + 4 * vi1);
    device atomic_uint* gp2_a = (device atomic_uint*)(gradPos + 4 * vi2);
#endif
    atomicAddFloat(&gp1_a[0], gp1x);
    atomicAddFloat(&gp1_a[1], gp1y);
    atomicAddFloat(&gp1_a[3], gp1w);
    atomicAddFloat(&gp2_a[0], gp2x);
    atomicAddFloat(&gp2_a[1], gp2y);
    atomicAddFloat(&gp2_a[3], gp2w);
}
