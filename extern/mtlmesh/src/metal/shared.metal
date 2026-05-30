// Utility compute kernels
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

kernel void arange_kernel(
    device int* array [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant int& stride [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    array[tid] = (int)tid * stride;
}

kernel void fill_int_kernel(
    device int* array [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant int& value [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    array[tid] = value;
}

kernel void fill_uint8_kernel(
    device uint8_t* array [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant uint8_t& value [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    array[tid] = value;
}

kernel void scatter_int_kernel(
    device const int* indices [[buffer(0)]],
    device const int* values [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device int* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[indices[tid]] = values[tid];
}

kernel void scatter_uint8_kernel(
    device const int* indices [[buffer(0)]],
    device const uint8_t* values [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device uint8_t* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[indices[tid]] = values[tid];
}

kernel void index_int_kernel(
    device const int* values [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device int* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = values[indices[tid]];
}

kernel void index_uint8_kernel(
    device const uint8_t* values [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device uint8_t* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = values[indices[tid]];
}

kernel void index_float_kernel(
    device const float* values [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = values[indices[tid]];
}

kernel void index_float3_kernel(
    device const packed_float3* values [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device packed_float3* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = values[indices[tid]];
}

kernel void diff_int_kernel(
    device const int* values [[buffer(0)]],
    constant int& N [[buffer(1)]],
    device int* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = values[tid + 1] - values[tid];
}

kernel void set_flag_kernel(
    device const int* indices [[buffer(0)]],
    constant int& N [[buffer(1)]],
    device int* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[indices[tid]] = 1;
}

// Cast kernel: int -> float
kernel void cast_int_to_float_kernel(
    device const int* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = float(input[tid]);
}

// Cast kernel: float -> int
kernel void cast_float_to_int_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = int(input[tid]);
}

// Compare kernel: less-than, output flag 1 if value < threshold
kernel void compare_less_than_float_kernel(
    device const float* values [[buffer(0)]],
    constant float& threshold [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device uint8_t* flag [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    flag[tid] = (values[tid] < threshold) ? 1 : 0;
}

// Compare kernel: greater-than-or-equal, output flag 1 if value >= threshold
kernel void compare_gte_float_kernel(
    device const float* values [[buffer(0)]],
    constant float& threshold [[buffer(1)]],
    constant int& N [[buffer(2)]],
    device uint8_t* flag [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    flag[tid] = (values[tid] >= threshold) ? 1 : 0;
}

// In-place division: a[i] = a[i] / float(b[i])
// Operates on packed_float3 (Vec3f) divided by int count — used in fill_holes centroid
kernel void inplace_div_float3_by_int_kernel(
    device packed_float3* a [[buffer(0)]],
    device const int* b [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    float3 val = float3(a[tid]);
    float divisor = float(b[tid]);
    a[tid] = packed_float3(val / divisor);
}

// In-place division: a[i] = a[i] / float(b[i]) for float arrays
kernel void inplace_div_float_by_int_kernel(
    device float* a [[buffer(0)]],
    device const int* b [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    a[tid] = a[tid] / float(b[tid]);
}

// Get diff kernel: mark transitions in sorted array (used by compress_ids)
// ids_diff[tid] = 1 if this is the last element or ids_sorted[tid] != ids_sorted[tid+1]
kernel void get_diff_int_kernel(
    device const int* ids_sorted [[buffer(0)]],
    device int* ids_diff [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    if (tid == (uint)N - 1) {
        ids_diff[tid] = 1;
        return;
    }
    ids_diff[tid] = (ids_sorted[tid] != ids_sorted[tid + 1]) ? 1 : 0;
}

// Copy Vec3f (16-byte aligned with padding) to packed_float3 (12-byte packed)
// Used in fill_holes to copy computed center vertices into vertex buffer
kernel void copy_vec3f_to_float3_kernel(
    device const float4* vec3f_data [[buffer(0)]],  // Vec3f is 16-byte aligned, treat as float4
    constant int& N [[buffer(1)]],
    device packed_float3* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    float4 v = vec3f_data[tid];
    output[tid] = packed_float3(float3(v.x, v.y, v.z));
}

// Fill float kernel
kernel void fill_float_kernel(
    device float* array [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant float& value [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    array[tid] = value;
}

// Fill uint kernel (used for propagated_cost_vals initialization)
kernel void fill_uint_kernel(
    device uint* array [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant uint& value [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    array[tid] = value;
}

// Union-Find: Hook edges
kernel void hook_edges_kernel(
    device const int2* adj [[buffer(0)]],
    constant int& M [[buffer(1)]],
    device atomic_int* conn_comp_ids [[buffer(2)]],
    device atomic_int* end_flag [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;

    int f0 = adj[tid].x;
    int f1 = adj[tid].y;

    // Find roots
    int root0 = atomic_load_explicit(&conn_comp_ids[f0], memory_order_relaxed);
    while (root0 != atomic_load_explicit(&conn_comp_ids[root0], memory_order_relaxed)) {
        root0 = atomic_load_explicit(&conn_comp_ids[root0], memory_order_relaxed);
    }
    int root1 = atomic_load_explicit(&conn_comp_ids[f1], memory_order_relaxed);
    while (root1 != atomic_load_explicit(&conn_comp_ids[root1], memory_order_relaxed)) {
        root1 = atomic_load_explicit(&conn_comp_ids[root1], memory_order_relaxed);
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomic_fetch_min_explicit(&conn_comp_ids[high], low, memory_order_relaxed);
    atomic_store_explicit(end_flag, 0, memory_order_relaxed);
}

// Union-Find: Compress components (path compression)
kernel void compress_components_kernel(
    device atomic_int* conn_comp_ids [[buffer(0)]],
    constant int& F [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;

    int p = atomic_load_explicit(&conn_comp_ids[tid], memory_order_relaxed);
    while (p != atomic_load_explicit(&conn_comp_ids[p], memory_order_relaxed)) {
        p = atomic_load_explicit(&conn_comp_ids[p], memory_order_relaxed);
    }
    atomic_store_explicit(&conn_comp_ids[tid], p, memory_order_relaxed);
}

// Remap faces after vertex compaction
kernel void remap_faces_kernel(
    device const int* vertices_map [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device packed_int3* faces [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 f = to_int3(faces[tid]);
    faces[tid] = from_int3(int3(vertices_map[f.x], vertices_map[f.y], vertices_map[f.z]));
}

// Set vertex as referenced
kernel void set_vertex_is_referenced_kernel(
    device const packed_int3* faces [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device atomic_int* vertex_is_referenced [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 face = to_int3(faces[tid]);
    atomic_store_explicit(&vertex_is_referenced[face.x], 1, memory_order_relaxed);
    atomic_store_explicit(&vertex_is_referenced[face.y], 1, memory_order_relaxed);
    atomic_store_explicit(&vertex_is_referenced[face.z], 1, memory_order_relaxed);
}

// Compress vertices using prefix sum map
kernel void compress_vertices_kernel(
    device const int* vertices_map [[buffer(0)]],
    device const packed_float3* old_vertices [[buffer(1)]],
    constant int& V [[buffer(2)]],
    device packed_float3* new_vertices [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}

// Compress faces using prefix sum map
kernel void compress_faces_kernel(
    device const int* faces_map [[buffer(0)]],
    device const int* vertices_map [[buffer(1)]],
    device const packed_int3* old_faces [[buffer(2)]],
    constant int& F [[buffer(3)]],
    device packed_int3* new_faces [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int new_id = faces_map[tid];
    int is_kept = faces_map[tid + 1] == new_id + 1;
    if (is_kept) {
        int3 f = to_int3(old_faces[tid]);
        new_faces[new_id] = from_int3(int3(vertices_map[f.x], vertices_map[f.y], vertices_map[f.z]));
    }
}
