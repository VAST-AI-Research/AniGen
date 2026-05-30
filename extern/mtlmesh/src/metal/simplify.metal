// Mesh simplification kernels - QEM-based edge collapse
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

// Compute QEM per vertex
kernel void get_qem_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    device const int* vert2face [[buffer(2)]],
    device const int* vert2face_offset [[buffer(3)]],
    constant int& V [[buffer(4)]],
    device QEM* qems [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)V) return;
    QEM v_qem;
    v_qem.zero();
    for (int f = vert2face_offset[tid]; f < vert2face_offset[tid+1]; f++) {
        int3 fv = to_int3(faces[vert2face[f]]);
        Vec3f f_v0 = Vec3f(float3(vertices[fv.x]));
        Vec3f e1 = Vec3f(float3(vertices[fv.y]));
        Vec3f e2 = Vec3f(float3(vertices[fv.z]));
        e1 = e1 - f_v0;
        e2 = e2 - f_v0;
        Vec3f n = e1.cross(e2);
        n.normalize();
        float d = -(n.dot(f_v0));
        v_qem.add_plane(float4(n.x, n.y, n.z, d));
    }
    qems[tid] = v_qem;
}

inline bool process_incident_tri(
    int tri_idx, int keep_vert, int other_vert,
    device const packed_float3* vertices, device const packed_int3* faces,
    thread const Vec3f& v_new, thread float& skinny_cost, thread int& num_tri
) {
    int3 fv = to_int3(faces[tri_idx]);
    if (fv.x == other_vert || fv.y == other_vert || fv.z == other_vert)
        return true; // shared face, will be removed

    Vec3f a = Vec3f(float3(vertices[fv.x]));
    Vec3f b = Vec3f(float3(vertices[fv.y]));
    Vec3f c = Vec3f(float3(vertices[fv.z]));

    Vec3f na = (fv.x == keep_vert) ? v_new : a;
    Vec3f nb = (fv.y == keep_vert) ? v_new : b;
    Vec3f nc = (fv.z == keep_vert) ? v_new : c;

    Vec3f old_normal = (b - a).cross(c - a);
    Vec3f new_e1 = nb - na, new_e2 = nc - na;
    Vec3f new_normal = new_e1.cross(new_e2);
    float new_area = 0.5f * new_normal.norm();

    if (old_normal.dot(new_normal) < 0.0f) return false; // flipped

    Vec3f new_e0 = nc - nb;
    float denom = new_e0.norm2() + new_e1.norm2() + new_e2.norm2();
    if (denom < 1e-12f) denom = 1e-12f;
    float shape = 4.0f * sqrt(3.0f) * new_area / denom;
    skinny_cost += 1.0f - clamp(shape, 0.0f, 1.0f);
    num_tri += 1;
    return true;
}

// Compute edge collapse cost
kernel void get_edge_collapse_cost_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    device const int* vert2face [[buffer(2)]],
    device const int* vert2face_offset [[buffer(3)]],
    device const ulong* edges [[buffer(4)]],
    device const uint8_t* vert_is_boundary [[buffer(5)]],
    device const QEM* qems [[buffer(6)]],
    constant int& E [[buffer(7)]],
    constant float& lambda_edge_length [[buffer(8)]],
    constant float& lambda_skinny [[buffer(9)]],
    device float* edge_collapse_costs [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    Vec3f v0 = Vec3f(float3(vertices[e0]));
    Vec3f v1 = Vec3f(float3(vertices[e1]));

    float w0 = 0.5f;
    if (vert_is_boundary[e0] && !vert_is_boundary[e1]) w0 = 1.0f;
    else if (!vert_is_boundary[e0] && vert_is_boundary[e1]) w0 = 0.0f;
    Vec3f v = v0 * w0 + v1 * (1.0f - w0);

    QEM q0 = qems[e0];
    QEM q1 = qems[e1];
    QEM edge_qem = q0 + q1;
    float cost = edge_qem.evaluate(v);

    float edge_length2 = (v1 - v0).norm2();
    cost += lambda_edge_length * edge_length2;

    float skinny_cost = 0.0f;
    int num_tri = 0;
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        if (!process_incident_tri(vert2face[f], e0, e1, vertices, faces, v, skinny_cost, num_tri)) {
            edge_collapse_costs[tid] = INFINITY;
            return;
        }
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        if (!process_incident_tri(vert2face[f], e1, e0, vertices, faces, v, skinny_cost, num_tri)) {
            edge_collapse_costs[tid] = INFINITY;
            return;
        }
    }
    if (num_tri > 0) skinny_cost /= num_tri;
    cost += lambda_skinny * skinny_cost * edge_length2;
    edge_collapse_costs[tid] = cost;
}

// Pack float cost (upper 32 bits) + edge id (lower 32 bits) into ulong.
// For positive floats, IEEE 754 bit order matches numeric order, so
// ulong comparison gives correct (cost, id) lexicographic min.
inline ulong pack_key_value_positive(int key, float value) {
    uint v = as_type<uint>(value);
    return (ulong(v) << 32) | ulong(uint(key));
}

// Propagate minimum edge collapse cost to neighboring faces.
// Single-pass 64-bit atomicMin.
kernel void propagate_cost_kernel(
    device const ulong* edges [[buffer(0)]],
    device const int* vert2face [[buffer(1)]],
    device const int* vert2face_offset [[buffer(2)]],
    device const float* edge_collapse_costs [[buffer(3)]],
    constant int& E [[buffer(4)]],
    device atomic_ulong* propagated_costs [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    ulong cost = pack_key_value_positive(int(tid), edge_collapse_costs[tid]);

    // S19-fix: keep memory_order_relaxed (MSL only supports relaxed); the
    // cross-dispatch ordering is enforced by the host command scheduler.
    // S22-verified: MSL compiler rejects seq_cst for atomic_min_explicit
    // on ulong ("order argument must be metal::memory_order_relaxed"), so
    // Codex's seq_cst hypothesis can't be directly tested in MSL.
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        atomic_min_explicit(&propagated_costs[vert2face[f]], cost, memory_order_relaxed);
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        atomic_min_explicit(&propagated_costs[vert2face[f]], cost, memory_order_relaxed);
    }
}

// Collapse edges in parallel (conflict-free via propagated cost check)
kernel void collapse_edges_kernel(
    device packed_float3* vertices [[buffer(0)]],
    device packed_int3* faces [[buffer(1)]],
    device const ulong* edges [[buffer(2)]],
    device const int* vert2face [[buffer(3)]],
    device const int* vert2face_offset [[buffer(4)]],
    device const float* edge_collapse_costs [[buffer(5)]],
    // S19-fix: declared `volatile` so the compiler cannot cache reads in
    // registers / tile L1.  Same physical buffer is bound as
    // `device atomic_ulong*` to propagate_cost_kernel (line 137).  Without
    // the volatile qualifier, the compiler may legally hoist or cache the
    // bare-load reads at lines 200+, causing the mutual-agreement check to
    // see stale UINT64_MAX or partial-update values instead of the final
    // reduced cost → simultaneous collapses on shared-vertex edges →
    // 92.6% NME explosion (S18 measurement).  MSL has no atomic_load on
    // ulong, so `volatile` is the portable forced-non-cached read.
    device const volatile ulong* propagated_costs [[buffer(6)]],
    device const uint8_t* vert_is_boundary [[buffer(7)]],
    constant int& E [[buffer(8)]],
    constant float& collapse_thresh [[buffer(9)]],
    device int* vertices_kept [[buffer(10)]],
    device int* faces_kept [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    float cost = edge_collapse_costs[tid];
    if (cost > collapse_thresh) return;

    ulong e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    ulong pack = pack_key_value_positive(int(tid), cost);

    // Check all neighboring faces agree this is their minimum-cost edge.
    // S19-fix: bare loads on a `volatile` pointer (declared above).  The
    // volatile qualifier forces the compiler to issue a fresh memory load
    // each time and disallows caching the value in a register / hoisting
    // out of the loop.
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }

    // Collapse: move e0 to midpoint, remove e1
    Vec3f v0 = Vec3f(float3(vertices[e0]));
    Vec3f v1 = Vec3f(float3(vertices[e1]));
    float w0 = 0.5f;
    if (vert_is_boundary[e0] && !vert_is_boundary[e1]) w0 = 1.0f;
    else if (!vert_is_boundary[e0] && vert_is_boundary[e1]) w0 = 0.0f;
    Vec3f v_new = v0 * w0 + v1 * (1.0f - w0);
    vertices[e0] = packed_float3(v_new.to_float3());
    vertices_kept[e1] = 0;

    // Remove shared faces, update remaining faces
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 fv = to_int3(faces[fid]);
        if (fv.x == e1 || fv.y == e1 || fv.z == e1)
            faces_kept[fid] = 0;
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int fid = vert2face[f];
        int3 fv = to_int3(faces[fid]);
        if (fv.x == e1) fv.x = e0;
        else if (fv.y == e1) fv.y = e0;
        else if (fv.z == e1) fv.z = e0;
        faces[fid] = from_int3(fv);
    }
}
