// Clean-up kernels: degenerate faces, duplicate detection, hole filling, orientation
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

// Sort vertices within each face for duplicate detection
kernel void sort_faces_kernel(
    device packed_int3* faces [[buffer(0)]],
    constant int& F [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 f = to_int3(faces[tid]);
    int tmp;
    if (f.x > f.y) { tmp = f.x; f.x = f.y; f.y = tmp; }
    if (f.y > f.z) { tmp = f.y; f.y = f.z; f.z = tmp; }
    if (f.x > f.y) { tmp = f.x; f.x = f.y; f.y = tmp; }
    faces[tid] = from_int3(f);
}

// Mark first face in each group of sorted duplicates
kernel void select_first_in_each_group_kernel(
    device const packed_int3* faces [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device uint8_t* face_mask [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    if (tid == 0) { face_mask[0] = 1; return; }
    packed_int3 f = faces[tid];
    packed_int3 p = faces[tid - 1];
    face_mask[tid] = (f.x == p.x && f.y == p.y && f.z == p.z) ? 0 : 1;
}

// Mark degenerate faces (duplicate verts or too thin)
kernel void mark_degenerate_faces_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    constant float& abs_thresh [[buffer(2)]],
    constant float& rel_thresh [[buffer(3)]],
    constant int& F [[buffer(4)]],
    device uint8_t* face_mask [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 face = to_int3(faces[tid]);

    if (face.x == face.y || face.y == face.z || face.z == face.x) {
        face_mask[tid] = 0; return;
    }

    Vec3f v0 = Vec3f(float3(vertices[face.x]));
    Vec3f v1 = Vec3f(float3(vertices[face.y]));
    Vec3f v2 = Vec3f(float3(vertices[face.z]));
    Vec3f e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;
    float max_edge = max(max(e0.norm(), e1.norm()), e2.norm());
    float area = e0.cross(e1).norm() * 0.5f;
    float thresh = min(rel_thresh * max_edge * max_edge, abs_thresh);
    face_mask[tid] = (area < thresh) ? 0 : 1;
}

// Mark non-manifold faces for removal
kernel void mark_non_manifold_faces_kernel(
    device const int* edge2face [[buffer(0)]],
    device const int* edge2face_offset [[buffer(1)]],
    device const int* edge2face_cnt [[buffer(2)]],
    constant int& E [[buffer(3)]],
    device uint8_t* face_keep_mask [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    int cnt = edge2face_cnt[tid];
    if (cnt <= 2) return;
    int start = edge2face_offset[tid];
    for (int i = 2; i < cnt; i++) {
        face_keep_mask[edge2face[start + i]] = 0;
    }
}

// Compute boundary edge lengths
kernel void compute_loop_boundary_lengths_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const ulong* edges [[buffer(1)]],
    device const int* loop_boundaries [[buffer(2)]],
    constant int& E [[buffer(3)]],
    device float* lengths [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(float3(vertices[e0]));
    Vec3f v1 = Vec3f(float3(vertices[e1]));
    lengths[tid] = (v1 - v0).norm();
}

// Compute midpoints for hole filling
kernel void compute_loop_boundary_midpoints_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const ulong* edges [[buffer(1)]],
    device const int* loop_boundaries [[buffer(2)]],
    constant int& E [[buffer(3)]],
    device packed_float3* midpoints [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(float3(vertices[e0]));
    Vec3f v1 = Vec3f(float3(vertices[e1]));
    Vec3f mid = (v0 + v1) * 0.5f;
    midpoints[tid] = packed_float3(mid.to_float3());
}

// Connect loop edges to new center vertex for hole filling
kernel void connect_new_vertices_kernel(
    device const ulong* edges [[buffer(0)]],
    device const int* loop_boundaries [[buffer(1)]],
    device const int* loop_bound_loop_ids [[buffer(2)]],
    constant int& L [[buffer(3)]],
    constant int& V [[buffer(4)]],
    device packed_int3* faces [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)L) return;
    int loop_id = loop_bound_loop_ids[tid];
    ulong e = edges[loop_boundaries[tid]];
    int e0 = int(e & 0xFFFFFFFF);
    int e1 = int(e >> 32);
    faces[tid] = from_int3(int3(e0, e1, loop_id + V));
}

// Get flip flags for face orientation unification
kernel void get_flip_flags_kernel(
    device const int2* manifold_face_adj [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    constant int& M [[buffer(2)]],
    device uint8_t* flipped [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;
    int2 adj_faces = manifold_face_adj[tid];
    int3 face1 = to_int3(faces[adj_faces.x]);
    int3 face2 = to_int3(faces[adj_faces.y]);

    int v1[3] = {face1.x, face1.y, face1.z};
    int si1[2] = {0, 0}, si2[2] = {0, 0};
    int found = 0;
    for (int i = 0; i < 3 && found < 2; i++) {
        if (v1[i] == face2.x) { si1[found] = i; si2[found] = 0; found++; }
        else if (v1[i] == face2.y) { si1[found] = i; si2[found] = 1; found++; }
        else if (v1[i] == face2.z) { si1[found] = i; si2[found] = 2; found++; }
    }
    int dir1 = (si1[1] - si1[0] + 3) % 3;
    int dir2 = (si2[1] - si2[0] + 3) % 3;
    flipped[tid] = (dir1 == dir2) ? 1 : 0;
}

// Hook edges with orientation tracking for unify_face_orientations
kernel void hook_edges_with_orientation_kernel(
    device const int2* adj [[buffer(0)]],
    device const uint8_t* flipped [[buffer(1)]],
    constant int& M [[buffer(2)]],
    device atomic_int* conn_comp_ids [[buffer(3)]],
    device atomic_int* end_flag [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;
    int f0 = adj[tid].x;
    int f1 = adj[tid].y;
    uint8_t is_flipped = flipped[tid];

    int val0 = atomic_load_explicit(&conn_comp_ids[f0], memory_order_relaxed);
    int root0 = val0 >> 1, flip0 = val0 & 1;
    while (true) {
        int pval = atomic_load_explicit(&conn_comp_ids[root0], memory_order_relaxed);
        int pr = pval >> 1;
        if (pr == root0) break;
        flip0 ^= pval & 1;
        root0 = pr;
    }

    int val1 = atomic_load_explicit(&conn_comp_ids[f1], memory_order_relaxed);
    int root1 = val1 >> 1, flip1 = val1 & 1;
    while (true) {
        int pval = atomic_load_explicit(&conn_comp_ids[root1], memory_order_relaxed);
        int pr = pval >> 1;
        if (pr == root1) break;
        flip1 ^= pval & 1;
        root1 = pr;
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomic_fetch_min_explicit(&conn_comp_ids[high], (low << 1) | (is_flipped ^ flip0 ^ flip1), memory_order_relaxed);
    atomic_store_explicit(end_flag, 0, memory_order_relaxed);
}

// Compress with orientation
kernel void compress_components_with_orientation_kernel(
    device atomic_int* conn_comp_ids [[buffer(0)]],
    constant int& F [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int val = atomic_load_explicit(&conn_comp_ids[tid], memory_order_relaxed);
    int p = val >> 1, f = val & 1;
    while (true) {
        int pval = atomic_load_explicit(&conn_comp_ids[p], memory_order_relaxed);
        int pr = pval >> 1;
        if (pr == p) break;
        f ^= pval & 1;
        p = pr;
    }
    atomic_store_explicit(&conn_comp_ids[tid], (p << 1) | f, memory_order_relaxed);
}

// Flip faces based on orientation flags
kernel void inplace_flip_faces_kernel(
    device packed_int3* faces [[buffer(0)]],
    device const int* conn_comp_with_flip [[buffer(1)]],
    constant int& F [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    if (conn_comp_with_flip[tid] & 1) {
        packed_int3 f = faces[tid];
        faces[tid] = from_int3(int3(f.x, f.z, f.y));
    }
}

// Construct vertex adjacency pairs for non-manifold repair
kernel void construct_vertex_adj_pairs_kernel(
    device const int2* manifold_face_adj [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    device int2* vertex_adj_pairs [[buffer(2)]],
    constant int& M [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;
    int2 af = manifold_face_adj[tid];
    int3 f1 = to_int3(faces[af.x]), f2 = to_int3(faces[af.y]);
    int v1[3] = {f1.x, f1.y, f1.z};
    int si1[2] = {0,0}, si2[2] = {0,0};
    int found = 0;
    for (int i = 0; i < 3 && found < 2; i++) {
        if (v1[i] == f2.x) { si1[found]=i; si2[found]=0; found++; }
        else if (v1[i] == f2.y) { si1[found]=i; si2[found]=1; found++; }
        else if (v1[i] == f2.z) { si1[found]=i; si2[found]=2; found++; }
    }
    if (found == 2) {
        vertex_adj_pairs[2*tid+0] = int2(3*af.x + si1[0], 3*af.y + si2[0]);
        vertex_adj_pairs[2*tid+1] = int2(3*af.x + si1[1], 3*af.y + si2[1]);
    } else {
        vertex_adj_pairs[2*tid+0] = int2(3*af.x, 3*af.x);
        vertex_adj_pairs[2*tid+1] = int2(3*af.y, 3*af.y);
    }
}

// Index vertices by face index for non-manifold repair
kernel void index_vertice_kernel(
    device const int* vertex_ids [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    device const packed_float3* vertices [[buffer(2)]],
    constant int& V [[buffer(3)]],
    device packed_float3* new_vertices [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)V) return;
    int vid = vertex_ids[tid];
    int3 face = to_int3(faces[vid / 3]);
    int f[3] = {face.x, face.y, face.z};
    new_vertices[tid] = vertices[f[vid % 3]];
}

// Copy int array to packed_int3 array
kernel void copy_int_to_int3_kernel(
    device const int* input [[buffer(0)]],
    constant int& N [[buffer(1)]],
    device packed_int3* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    output[tid] = from_int3(int3(input[3*tid], input[3*tid+1], input[3*tid+2]));
}
