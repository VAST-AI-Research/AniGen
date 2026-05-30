// Metal differentiable rasterization kernel.
// Uses a Metal render pipeline (hardware rasterization) for UV-space triangle rasterization
// Outputs: per-pixel (face_id, u, v, depth) as RGBA32Float
#include <metal_stdlib>
using namespace metal;

// Triangle ID <-> float32 conversion.
// Values <= 16777216 (2^24) are stored as plain float cast.
// Larger values use a biased bitwise mapping to avoid precision loss.
// Max supported: 889192447.
inline float triidx_to_float(int x) {
    if (x <= 0x01000000) return float(x);
    return as_type<float>(0x4a800000 + x);
}
inline int float_to_triidx(float x) {
    if (x <= 16777216.0f) return int(x);
    return as_type<int>(x) - 0x4a800000;
}

// ========== Render Pipeline (Hardware Rasterization) ==========

struct VertexIn {
    float4 position [[attribute(0)]];  // clip-space position (from UV coords)
    uint vertex_id [[attribute(1)]];
};

struct RasterizerData {
    float4 position [[position]];
    float3 barycentrics;   // Will be computed from edge functions in fragment
    uint triangle_id;
    float2 uv;
};

// Vertex shader: pass UV coordinates as clip-space positions
// For UV-space rasterization: pos.xy = uv * 2 - 1 (NDC), pos.z = depth, pos.w = 1
vertex RasterizerData rasterize_vertex(
    const device float4* positions [[buffer(0)]],
    const device int* triangles [[buffer(1)]],
    constant int& num_vertices [[buffer(2)]],
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]]
) {
    RasterizerData out;

    // Triangle and vertex indices
    uint tri_id = vertex_id / 3;
    uint local_id = vertex_id % 3;
    int vi = triangles[tri_id * 3 + local_id];

    float4 pos = positions[vi];
    out.position = pos;
    // nvdiffrast uses OpenGL clip space (z in [-w, w]); Metal's clip volume is
    // [0, w].  Remap GL clip z -> Metal clip z so (a) the full NDC z range
    // [-1, 1] survives near-plane clipping and (b) a smaller z/w (nvdiffrast's
    // "closer", which must win the depth test) maps to a smaller Metal depth.
    out.position.z = (pos.z + pos.w) * 0.5f;
    out.triangle_id = tri_id;

    return out;
}

// Fragment shader: output (u, v, z/w, triangle_id) per pixel
// Barycentrics come from the rasterizer's built-in interpolation
fragment float4 rasterize_fragment(
    RasterizerData in [[stage_in]],
    float3 barycentrics [[barycentric_coord]]
) {
    float u = barycentrics.x;
    float v = barycentrics.y;
    // in.position.z is the Metal window depth in [0,1] = (z/w + 1)/2 after the
    // vertex z-remap.  Recover nvdiffrast's NDC z/w in [-1,1] for the output.
    float zw = 2.0f * in.position.z - 1.0f;

    // Clamp barycentrics to [0, 1]
    u = saturate(u);
    v = saturate(v);
    float bs = 1.0f / max(u + v, 1.0f);
    u *= bs;
    v *= bs;
    zw = clamp(zw, -1.0f, 1.0f);

    // Encode triangle ID as float (supports up to ~16M triangles with float32)
    // Use +1 offset so 0 means "no triangle"
    float tri_id_float = triidx_to_float(int(in.triangle_id + 1));

    return float4(u, v, zw, tri_id_float);
}

// ========== Compute Shader Fallback (for when render pipeline isn't available) ==========
// Software rasterization as a Metal compute shader
// Processes triangles in chunks, writing to a 2D output buffer

struct RasterizeParams {
    int num_triangles;
    int num_vertices;
    int width;
    int height;
    float xs, xo, ys, yo;  // pixel to clip-space transform
    int chunk_start;        // global triangle offset for chunked dispatch
};

kernel void rasterize_compute_kernel(
    device const float4* positions [[buffer(0)]],
    device const int* triangles [[buffer(1)]],
    constant RasterizeParams& params [[buffer(2)]],
    device float4* output [[buffer(3)]],     // [H, W] rast output
    device float4* output_db [[buffer(4)]],  // [H, W] bary derivatives
    uint2 gid [[thread_position_in_grid]]
) {
    int px = gid.x;
    int py = gid.y;
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    float fx = params.xs * float(px) + params.xo;
    float fy = params.ys * float(py) + params.yo;

    // Find closest triangle (simple scan — in practice, use the render pipeline)
    // nvdiffrast keeps the SMALLEST z/w (cudaraster atomicMin); init to the far
    // sentinel and accept strictly-smaller z/w so ties keep the first triangle.
    float best_z = 2.0f;
    float best_u = 0, best_v = 0;
    int best_tri = -1;
    float best_dudx = 0, best_dudy = 0, best_dvdx = 0, best_dvdy = 0;

    for (int tri = 0; tri < params.num_triangles; tri++) {
        int vi0 = triangles[tri * 3 + 0];
        int vi1 = triangles[tri * 3 + 1];
        int vi2 = triangles[tri * 3 + 2];

        if (vi0 < 0 || vi0 >= params.num_vertices ||
            vi1 < 0 || vi1 >= params.num_vertices ||
            vi2 < 0 || vi2 >= params.num_vertices) continue;

        float4 p0 = positions[vi0];
        float4 p1 = positions[vi1];
        float4 p2 = positions[vi2];

        // Edge functions
        float p0x = p0.x - fx * p0.w;
        float p0y = p0.y - fy * p0.w;
        float p1x = p1.x - fx * p1.w;
        float p1y = p1.y - fy * p1.w;
        float p2x = p2.x - fx * p2.w;
        float p2y = p2.y - fy * p2.w;
        float a0 = p1x*p2y - p1y*p2x;
        float a1 = p2x*p0y - p2y*p0x;
        float a2 = p0x*p1y - p0y*p1x;

        float at = a0 + a1 + a2;
        if (at <= 0) continue; // Back-face or degenerate

        float iw = 1.0f / at;
        float u = a0 * iw;
        float v = a1 * iw;

        // Depth
        float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
        float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
        float zw = z / w;

        // Inside triangle check
        if (u >= 0 && v >= 0 && (u + v) <= 1.0f && zw < best_z) {
            best_z = zw;

            // Bary pixel differentials — MUST use pre-clamp u,v
            float dfxdx = params.xs * iw;
            float dfydy = params.ys * iw;
            float da0dx = p2.y*p1.w - p1.y*p2.w;
            float da0dy = p1.x*p2.w - p2.x*p1.w;
            float da1dx = p0.y*p2.w - p2.y*p0.w;
            float da1dy = p2.x*p0.w - p0.x*p2.w;
            float da2dx = p1.y*p0.w - p0.y*p1.w;
            float da2dy = p0.x*p1.w - p1.x*p0.w;
            float datdx = da0dx + da1dx + da2dx;
            float datdy = da0dy + da1dy + da2dy;
            best_dudx = dfxdx * (u * datdx - da0dx);
            best_dudy = dfydy * (u * datdy - da0dy);
            best_dvdx = dfxdx * (v * datdx - da1dx);
            best_dvdy = dfydy * (v * datdy - da1dy);

            // Clamp barycentrics for output (after computing differentials)
            best_u = saturate(u);
            best_v = saturate(v);
            float bs = 1.0f / max(best_u + best_v, 1.0f);
            best_u *= bs;
            best_v *= bs;
            best_tri = tri;
        }
    }

    if (best_tri >= 0) {
        float tri_float = triidx_to_float(best_tri + params.chunk_start + 1);
        output[pidx] = float4(best_u, best_v, clamp(best_z, -1.0f, 1.0f), tri_float);
        output_db[pidx] = float4(best_dudx, best_dudy, best_dvdx, best_dvdy);
    } else {
        output[pidx] = float4(0);
        output_db[pidx] = float4(0);
    }
}

// ========== Bary Derivatives Compute Pass ==========
// Computes bary pixel differentials from rast_out (for hardware render pipeline path).
// For each covered pixel, reads the triangle ID, recomputes edge functions, and outputs derivatives.

kernel void rasterize_db_kernel(
    device const float4* positions [[buffer(0)]],
    device const int* triangles [[buffer(1)]],
    constant RasterizeParams& params [[buffer(2)]],
    device const float4* rast_in [[buffer(3)]],   // [H, W] rast output from render pass
    device float4* output_db [[buffer(4)]],        // [H, W] bary derivatives output
    uint2 gid [[thread_position_in_grid]]
) {
    int px = gid.x;
    int py = gid.y;
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    float4 rast = rast_in[pidx];

    int tri_idx = float_to_triidx(rast.w) - 1;
    if (tri_idx < 0 || tri_idx >= params.num_triangles) {
        output_db[pidx] = float4(0);
        return;
    }

    int vi0 = triangles[tri_idx * 3 + 0];
    int vi1 = triangles[tri_idx * 3 + 1];
    int vi2 = triangles[tri_idx * 3 + 2];

    float4 p0 = positions[vi0];
    float4 p1 = positions[vi1];
    float4 p2 = positions[vi2];

    float fx = params.xs * float(px) + params.xo;
    float fy = params.ys * float(py) + params.yo;

    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;

    float at = a0 + a1 + (p0x*p1y - p0y*p1x);
    if (abs(at) < 1e-12f) {
        output_db[pidx] = float4(0);
        return;
    }
    float iw = 1.0f / at;
    float u = a0 * iw;
    float v = a1 * iw;

    float dfxdx = params.xs * iw;
    float dfydy = params.ys * iw;
    float da0dx = p2.y*p1.w - p1.y*p2.w;
    float da0dy = p1.x*p2.w - p2.x*p1.w;
    float da1dx = p0.y*p2.w - p2.y*p0.w;
    float da1dy = p2.x*p0.w - p0.x*p2.w;
    float da2dx = p1.y*p0.w - p0.y*p1.w;
    float da2dy = p0.x*p1.w - p1.x*p0.w;
    float datdx = da0dx + da1dx + da2dx;
    float datdy = da0dy + da1dy + da2dy;

    float dudx = dfxdx * (u * datdx - da0dx);
    float dudy = dfydy * (u * datdy - da0dy);
    float dvdx = dfxdx * (v * datdx - da1dx);
    float dvdy = dfydy * (v * datdy - da1dy);

    output_db[pidx] = float4(dudx, dudy, dvdx, dvdy);
}

// ========== Backward Pass ==========
// Atomic float add via CAS loop (Metal 3.0 has no native atomic<float>).
inline void rast_atomicAddFloat(device atomic_uint* addr, float value)
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

// Rasterize gradient kernel — computes position gradients from bary gradients.
// Backward pass for differentiable rasterization (Laine et al. 2020, Section 3).
struct RasterizeGradParams {
    int num_triangles;
    int num_vertices;
    int width;
    int height;
    float xs, xo, ys, yo;
    int enable_db;
};

kernel void rasterize_grad_kernel(
    device const float4* pos        [[buffer(0)]],  // [V] float4
    device const int*    tri        [[buffer(1)]],  // [T*3] int
    device const float*  rast_out   [[buffer(2)]],  // [H, W, 4] float
    device const float*  dy         [[buffer(3)]],  // [H, W, 4] float (only .xy used)
    device const float*  ddb        [[buffer(4)]],  // [H, W, 4] float (bary deriv grads)
    device float*        grad       [[buffer(5)]],  // [V, 4] float (atomic accum)
    constant RasterizeGradParams& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= p.width || py >= p.height)
        return;

    int pidx = px + p.width * py;

    // Read triangle idx and dy.
    float dy_x = dy[pidx * 4 + 0];
    float dy_y = dy[pidx * 4 + 1];
    float4 ddb_v = p.enable_db ? float4(ddb[pidx * 4 + 0], ddb[pidx * 4 + 1],
                                         ddb[pidx * 4 + 2], ddb[pidx * 4 + 3])
                                : float4(0.f);
    int triIdx = float_to_triidx(rast_out[pidx * 4 + 3]) - 1;

    if (triIdx < 0 || triIdx >= p.num_triangles)
        return;

    // Quick zero-gradient check.
    int grad_all_dy = as_type<int>(dy_x) | as_type<int>(dy_y);
    int grad_all_ddb = 0;
    if (p.enable_db)
        grad_all_ddb = as_type<int>(ddb_v.x) | as_type<int>(ddb_v.y) |
                       as_type<int>(ddb_v.z) | as_type<int>(ddb_v.w);
    if (((grad_all_dy | grad_all_ddb) << 1) == 0)
        return;

    // Fetch vertex indices.
    int vi0 = tri[triIdx * 3 + 0];
    int vi1 = tri[triIdx * 3 + 1];
    int vi2 = tri[triIdx * 3 + 2];

    if (vi0 < 0 || vi0 >= p.num_vertices ||
        vi1 < 0 || vi1 >= p.num_vertices ||
        vi2 < 0 || vi2 >= p.num_vertices)
        return;

    // Fetch vertex positions.
    float4 p0 = pos[vi0];
    float4 p1 = pos[vi1];
    float4 p2 = pos[vi2];

    // Evaluate edge functions.
    float fx = p.xs * float(px) + p.xo;
    float fy = p.ys * float(py) + p.yo;
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;
    float a2 = p0x*p1y - p0y*p1x;

    // Inverse area with epsilon.
    float at = a0 + a1 + a2;
    float ep = copysign(1e-6f, at);
    float iw = 1.f / (at + ep);

    // Barycentrics.
    float b0 = a0 * iw;
    float b1 = a1 * iw;

    // Position gradients.
    float gb0  = dy_x * iw;
    float gb1  = dy_y * iw;
    float gbb  = gb0 * b0 + gb1 * b1;
    float gp0x = gbb * (p2y - p1y) - gb1 * p2y;
    float gp1x = gbb * (p0y - p2y) + gb0 * p2y;
    float gp2x = gbb * (p1y - p0y) - gb0 * p1y + gb1 * p0y;
    float gp0y = gbb * (p1x - p2x) + gb1 * p2x;
    float gp1y = gbb * (p2x - p0x) - gb0 * p2x;
    float gp2y = gbb * (p0x - p1x) + gb0 * p1x - gb1 * p0x;
    float gp0w = -fx * gp0x - fy * gp0y;
    float gp1w = -fx * gp1x - fy * gp1y;
    float gp2w = -fx * gp2x - fy * gp2y;

    // Bary differential gradients.
    if (p.enable_db && ((grad_all_ddb) << 1) != 0)
    {
        float dfxdX = p.xs * iw;
        float dfydY = p.ys * iw;
        ddb_v.x *= dfxdX;
        ddb_v.y *= dfydY;
        ddb_v.z *= dfxdX;
        ddb_v.w *= dfydY;

        float da0dX = p1.y * p2.w - p2.y * p1.w;
        float da1dX = p2.y * p0.w - p0.y * p2.w;
        float da2dX = p0.y * p1.w - p1.y * p0.w;
        float da0dY = p2.x * p1.w - p1.x * p2.w;
        float da1dY = p0.x * p2.w - p2.x * p0.w;
        float da2dY = p1.x * p0.w - p0.x * p1.w;
        float datdX = da0dX + da1dX + da2dX;
        float datdY = da0dY + da1dY + da2dY;

        float x01 = p0.x - p1.x;
        float x12 = p1.x - p2.x;
        float x20 = p2.x - p0.x;
        float y01 = p0.y - p1.y;
        float y12 = p1.y - p2.y;
        float y20 = p2.y - p0.y;
        float w01 = p0.w - p1.w;
        float w12 = p1.w - p2.w;
        float w20 = p2.w - p0.w;

        float a0p1 = fy * p2.x - fx * p2.y;
        float a0p2 = fx * p1.y - fy * p1.x;
        float a1p0 = fx * p2.y - fy * p2.x;
        float a1p2 = fy * p0.x - fx * p0.y;

        float wdudX = 2.f * b0 * datdX - da0dX;
        float wdudY = 2.f * b0 * datdY - da0dY;
        float wdvdX = 2.f * b1 * datdX - da1dX;
        float wdvdY = 2.f * b1 * datdY - da1dY;

        float c0  = iw * (ddb_v.x * wdudX + ddb_v.y * wdudY + ddb_v.z * wdvdX + ddb_v.w * wdvdY);
        float cx  = c0 * fx - ddb_v.x * b0 - ddb_v.z * b1;
        float cy  = c0 * fy - ddb_v.y * b0 - ddb_v.w * b1;
        float cxy = iw * (ddb_v.x * datdX + ddb_v.y * datdY);
        float czw = iw * (ddb_v.z * datdX + ddb_v.w * datdY);

        gp0x += c0 * y12 - cy * w12              + czw * p2y                                               + ddb_v.w * p2.w;
        gp1x += c0 * y20 - cy * w20 - cxy * p2y                              - ddb_v.y * p2.w;
        gp2x += c0 * y01 - cy * w01 + cxy * p1y  - czw * p0y                 + ddb_v.y * p1.w                - ddb_v.w * p0.w;
        gp0y += cx * w12 - c0 * x12              - czw * p2x                                - ddb_v.z * p2.w;
        gp1y += cx * w20 - c0 * x20 + cxy * p2x               + ddb_v.x * p2.w;
        gp2y += cx * w01 - c0 * x01 - cxy * p1x  + czw * p0x  - ddb_v.x * p1.w                + ddb_v.z * p0.w;
        gp0w += cy * x12 - cx * y12              - czw * a1p0                               + ddb_v.z * p2.y - ddb_v.w * p2.x;
        gp1w += cy * x20 - cx * y20 - cxy * a0p1              - ddb_v.x * p2.y + ddb_v.y * p2.x;
        gp2w += cy * x01 - cx * y01 - cxy * a0p2 - czw * a1p2 + ddb_v.x * p1.y - ddb_v.y * p1.x - ddb_v.z * p0.y + ddb_v.w * p0.x;
    }

    // Atomic accumulate to grad.
    device atomic_uint* g0 = (device atomic_uint*)(grad + 4 * vi0);
    device atomic_uint* g1 = (device atomic_uint*)(grad + 4 * vi1);
    device atomic_uint* g2 = (device atomic_uint*)(grad + 4 * vi2);
    rast_atomicAddFloat(&g0[0], gp0x);
    rast_atomicAddFloat(&g0[1], gp0y);
    rast_atomicAddFloat(&g0[3], gp0w);
    rast_atomicAddFloat(&g1[0], gp1x);
    rast_atomicAddFloat(&g1[1], gp1y);
    rast_atomicAddFloat(&g1[3], gp1w);
    rast_atomicAddFloat(&g2[0], gp2x);
    rast_atomicAddFloat(&g2[1], gp2y);
    rast_atomicAddFloat(&g2[3], gp2w);
}
