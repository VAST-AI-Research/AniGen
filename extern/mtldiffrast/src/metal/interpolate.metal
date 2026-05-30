// Metal interpolation kernel — forward pass for vertex attribute interpolation.
#include <metal_stdlib>
using namespace metal;

// Triangle ID <-> float32 conversion.
// Must be kept in sync with rasterize.metal triidx_to_float.
inline int float_to_triidx(float x) {
    if (x <= 16777216.0f) return int(x);
    return as_type<int>(x) - 0x4a800000;
}

struct InterpolateParams {
    int num_triangles;
    int num_vertices;
    int num_attr;
    int width;
    int height;
};

// Forward interpolation: given rast output (u,v,z/w,tri_id) + vertex attributes,
// compute per-pixel interpolated attributes
kernel void interpolate_fwd_kernel(
    device const int* triangles [[buffer(0)]],       // [T*3] int
    device const float* attributes [[buffer(1)]],    // [V, A] float
    device const float4* rast [[buffer(2)]],         // [H, W] rast output
    constant InterpolateParams& params [[buffer(3)]],
    device float* output [[buffer(4)]],              // [H, W, A] float
    uint2 gid [[thread_position_in_grid]]
) {
    int px = gid.x;
    int py = gid.y;
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    float4 r = rast[pidx];
    int triIdx = float_to_triidx(r.w) - 1;

    int A = params.num_attr;
    int base = pidx * A;

    if (triIdx < 0 || triIdx >= params.num_triangles) {
        for (int i = 0; i < A; i++) output[base + i] = 0.0f;
        return;
    }

    // Fetch vertex indices
    int vi0 = triangles[triIdx * 3 + 0];
    int vi1 = triangles[triIdx * 3 + 1];
    int vi2 = triangles[triIdx * 3 + 2];

    if (vi0 < 0 || vi0 >= params.num_vertices ||
        vi1 < 0 || vi1 >= params.num_vertices ||
        vi2 < 0 || vi2 >= params.num_vertices) {
        for (int i = 0; i < A; i++) output[base + i] = 0.0f;
        return;
    }

    // Barycentrics
    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.0f - b0 - b1;

    // Interpolate all attributes
    device const float* a0 = attributes + vi0 * A;
    device const float* a1 = attributes + vi1 * A;
    device const float* a2 = attributes + vi2 * A;

    for (int i = 0; i < A; i++) {
        output[base + i] = b0 * a0[i] + b1 * a1[i] + b2 * a2[i];
    }
}

// Forward interpolation with pixel differentials
kernel void interpolate_fwd_da_kernel(
    device const int* triangles [[buffer(0)]],
    device const float* attributes [[buffer(1)]],
    device const float4* rast [[buffer(2)]],
    device const float4* rast_db [[buffer(3)]],      // [H, W] bary derivatives
    constant InterpolateParams& params [[buffer(4)]],
    device float* output [[buffer(5)]],              // [H, W, A]
    device float2* output_da [[buffer(6)]],          // [H, W, A] pixel derivatives
    uint2 gid [[thread_position_in_grid]]
) {
    int px = gid.x;
    int py = gid.y;
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    float4 r = rast[pidx];
    int triIdx = float_to_triidx(r.w) - 1;

    int A = params.num_attr;
    int base = pidx * A;

    if (triIdx < 0 || triIdx >= params.num_triangles) {
        for (int i = 0; i < A; i++) {
            output[base + i] = 0.0f;
            output_da[base + i] = float2(0);
        }
        return;
    }

    int vi0 = triangles[triIdx * 3 + 0];
    int vi1 = triangles[triIdx * 3 + 1];
    int vi2 = triangles[triIdx * 3 + 2];

    if (vi0 < 0 || vi0 >= params.num_vertices ||
        vi1 < 0 || vi1 >= params.num_vertices ||
        vi2 < 0 || vi2 >= params.num_vertices) {
        for (int i = 0; i < A; i++) {
            output[base + i] = 0.0f;
            output_da[base + i] = float2(0);
        }
        return;
    }

    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.0f - b0 - b1;

    device const float* a0 = attributes + vi0 * A;
    device const float* a1 = attributes + vi1 * A;
    device const float* a2 = attributes + vi2 * A;

    // Interpolate
    for (int i = 0; i < A; i++) {
        output[base + i] = b0 * a0[i] + b1 * a1[i] + b2 * a2[i];
    }

    // Pixel differentials
    float4 db = rast_db[pidx];
    float dudx = db.x, dudy = db.y, dvdx = db.z, dvdy = db.w;

    for (int i = 0; i < A; i++) {
        float dsdu = a0[i] - a2[i];
        float dsdv = a1[i] - a2[i];
        float dsdx = dudx * dsdu + dvdx * dsdv;
        float dsdy = dudy * dsdu + dvdy * dsdv;
        output_da[base + i] = float2(dsdx, dsdy);
    }
}

// ========== Backward Pass ==========

// Atomic float add via CAS loop.
inline void ip_atomicAddFloat(device atomic_uint* addr, float value)
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

struct InterpolateGradParams {
    int num_triangles;
    int num_vertices;
    int num_attr;
    int width;
    int height;
};

// Interpolate gradient kernel (no pixel derivatives).
// Computes gradients w.r.t. vertex attributes and rasterizer output.
kernel void interpolate_grad_kernel(
    device const int*    triangles   [[buffer(0)]],  // [T*3]
    device const float*  attributes  [[buffer(1)]],  // [V, A]
    device const float4* rast        [[buffer(2)]],  // [H, W]
    device const float*  dy          [[buffer(3)]],  // [H, W, A]
    device float*        grad_attr   [[buffer(4)]],  // [V, A] (atomic accum)
    device float*        grad_rast   [[buffer(5)]],  // [H, W, 4]
    constant InterpolateGradParams& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= p.width || py >= p.height)
        return;

    int pidx = px + p.width * py;
    int A = p.num_attr;

    // Fetch triangle ID.
    float4 r = rast[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    if (triIdx < 0 || triIdx >= p.num_triangles) {
        // Zero grad_rast for this pixel.
        ((device float4*)grad_rast)[pidx] = float4(0.f);
        return;
    }

    // Fetch vertex indices.
    int vi0 = triangles[triIdx * 3 + 0];
    int vi1 = triangles[triIdx * 3 + 1];
    int vi2 = triangles[triIdx * 3 + 2];

    if (vi0 < 0 || vi0 >= p.num_vertices ||
        vi1 < 0 || vi1 >= p.num_vertices ||
        vi2 < 0 || vi2 >= p.num_vertices)
        return;

    device const float* a0 = attributes + vi0 * A;
    device const float* a1 = attributes + vi1 * A;
    device const float* a2 = attributes + vi2 * A;
    device const float* pdy = dy + pidx * A;

    // Barycentrics.
    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.f - b0 - b1;
    float gb0 = 0.f;
    float gb1 = 0.f;

    // Loop over attributes.
    for (int i = 0; i < A; i++) {
        float y = pdy[i];
        float s0 = a0[i];
        float s1 = a1[i];
        float s2 = a2[i];
        gb0 += y * (s0 - s2);
        gb1 += y * (s1 - s2);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi0 * A + i), b0 * y);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi1 * A + i), b1 * y);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi2 * A + i), b2 * y);
    }

    // Write bary gradients.
    ((device float4*)grad_rast)[pidx] = float4(gb0, gb1, 0.f, 0.f);
}

// Interpolate gradient kernel with pixel derivatives (DA mode).
kernel void interpolate_grad_da_kernel(
    device const int*    triangles   [[buffer(0)]],  // [T*3]
    device const float*  attributes  [[buffer(1)]],  // [V, A]
    device const float4* rast        [[buffer(2)]],  // [H, W]
    device const float4* rast_db     [[buffer(3)]],  // [H, W]
    device const float*  dy          [[buffer(4)]],  // [H, W, A]
    device const float2* dda         [[buffer(5)]],  // [H, W, A] pixel diff grads
    device float*        grad_attr   [[buffer(6)]],  // [V, A] (atomic accum)
    device float*        grad_rast   [[buffer(7)]],  // [H, W, 4]
    device float*        grad_rast_db [[buffer(8)]], // [H, W, 4]
    constant InterpolateGradParams& p [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= p.width || py >= p.height)
        return;

    int pidx = px + p.width * py;
    int A = p.num_attr;

    float4 r = rast[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    if (triIdx < 0 || triIdx >= p.num_triangles) {
        ((device float4*)grad_rast)[pidx] = float4(0.f);
        ((device float4*)grad_rast_db)[pidx] = float4(0.f);
        return;
    }

    int vi0 = triangles[triIdx * 3 + 0];
    int vi1 = triangles[triIdx * 3 + 1];
    int vi2 = triangles[triIdx * 3 + 2];

    if (vi0 < 0 || vi0 >= p.num_vertices ||
        vi1 < 0 || vi1 >= p.num_vertices ||
        vi2 < 0 || vi2 >= p.num_vertices)
        return;

    device const float* a0 = attributes + vi0 * A;
    device const float* a1 = attributes + vi1 * A;
    device const float* a2 = attributes + vi2 * A;
    device const float* pdy = dy + pidx * A;

    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.f - b0 - b1;
    float gb0 = 0.f;
    float gb1 = 0.f;

    // Attribute gradients from interpolation.
    for (int i = 0; i < A; i++) {
        float y = pdy[i];
        float s0 = a0[i];
        float s1 = a1[i];
        float s2 = a2[i];
        gb0 += y * (s0 - s2);
        gb1 += y * (s1 - s2);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi0 * A + i), b0 * y);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi1 * A + i), b1 * y);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi2 * A + i), b2 * y);
    }

    ((device float4*)grad_rast)[pidx] = float4(gb0, gb1, 0.f, 0.f);

    // Pixel derivative gradients.
    device const float2* pdda = dda + pidx * A;
    float gdudx = 0.f, gdudy = 0.f, gdvdx = 0.f, gdvdy = 0.f;

    float4 db = rast_db[pidx];
    float dudx = db.x, dudy = db.y, dvdx = db.z, dvdy = db.w;

    // All attrs are diff attrs (diff_attrs_all = true).
    for (int i = 0; i < A; i++) {
        float2 dsdxy = pdda[i];
        float dsdx = dsdxy.x;
        float dsdy = dsdxy.y;

        float s0 = a0[i];
        float s1 = a1[i];
        float s2 = a2[i];

        float dsdu = s0 - s2;
        float dsdv = s1 - s2;
        gdudx += dsdu * dsdx;
        gdudy += dsdu * dsdy;
        gdvdx += dsdv * dsdx;
        gdvdy += dsdv * dsdy;

        float du = dsdx * dudx + dsdy * dudy;
        float dv = dsdx * dvdx + dsdy * dvdy;
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi0 * A + i), du);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi1 * A + i), dv);
        ip_atomicAddFloat((device atomic_uint*)(grad_attr + vi2 * A + i), -du - dv);
    }

    ((device float4*)grad_rast_db)[pidx] = float4(gdudx, gdudy, gdvdx, gdvdy);
}
