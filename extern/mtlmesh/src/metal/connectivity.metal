// Connectivity compute kernels - edges, adjacency, boundaries
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

// Count neighbor faces per vertex (atomics)
kernel void get_neighbor_face_cnt_kernel(
    device const packed_int3* faces [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device atomic_int* neighbor_face_cnt [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 f = to_int3(faces[tid]);
    atomic_fetch_add_explicit(&neighbor_face_cnt[f.x], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&neighbor_face_cnt[f.y], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&neighbor_face_cnt[f.z], 1, memory_order_relaxed);
}

// Fill neighbor face IDs using atomic counters for placement
kernel void fill_neighbor_face_ids_kernel(
    device const packed_int3* faces [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device int* neighbor_face_ids [[buffer(2)]],
    device const int* neighbor_face_ids_offset [[buffer(3)]],
    device atomic_int* neighbor_face_cnt [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 f = to_int3(faces[tid]);
    int idx;
    idx = atomic_fetch_add_explicit(&neighbor_face_cnt[f.x], 1, memory_order_relaxed);
    neighbor_face_ids[neighbor_face_ids_offset[f.x] + idx] = (int)tid;
    idx = atomic_fetch_add_explicit(&neighbor_face_cnt[f.y], 1, memory_order_relaxed);
    neighbor_face_ids[neighbor_face_ids_offset[f.y] + idx] = (int)tid;
    idx = atomic_fetch_add_explicit(&neighbor_face_cnt[f.z], 1, memory_order_relaxed);
    neighbor_face_ids[neighbor_face_ids_offset[f.z] + idx] = (int)tid;
}

// Expand 3 edges per face, encode as uint64 (min_vert << 32 | max_vert)
kernel void expand_edges_kernel(
    device const packed_int3* faces [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device ulong* edges [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int base = (int)tid * 3;
    int3 f = to_int3(faces[tid]);
    edges[base + 0] = ((ulong)min(f.x, f.y) << 32) | (ulong)max(f.x, f.y);
    edges[base + 1] = ((ulong)min(f.y, f.z) << 32) | (ulong)max(f.y, f.z);
    edges[base + 2] = ((ulong)min(f.z, f.x) << 32) | (ulong)max(f.z, f.x);
}

// Edge-face adjacency kernel
kernel void get_edge_face_adjacency_kernel(
    device const packed_int3* faces [[buffer(0)]],
    device const ulong* edges [[buffer(1)]],
    device const int* edge2face_cnt [[buffer(2)]],
    device const int* vert2face [[buffer(3)]],
    device const int* vert2face_offset [[buffer(4)]],
    device const int* edge2face_offset [[buffer(5)]],
    constant int& E [[buffer(6)]],
    device int* edge2face [[buffer(7)]],
    device packed_int3* face2edge [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    int ptr = edge2face_offset[tid];

    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 fv = to_int3(faces[fid]);
        if (fv.x == e1 || fv.y == e1 || fv.z == e1) {
            edge2face[ptr++] = fid;
            if ((fv.x == e0 && fv.y == e1) || (fv.x == e1 && fv.y == e0))
                face2edge[fid].x = (int)tid;
            else if ((fv.y == e0 && fv.z == e1) || (fv.y == e1 && fv.z == e0))
                face2edge[fid].y = (int)tid;
            else if ((fv.z == e0 && fv.x == e1) || (fv.z == e1 && fv.x == e0))
                face2edge[fid].z = (int)tid;
        }
    }
}

// Vertex-edge adjacency: count
kernel void get_vertex_edge_cnt_kernel(
    device const ulong* edges [[buffer(0)]],
    constant int& E [[buffer(1)]],
    device atomic_int* vert2edge_cnt [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    atomic_fetch_add_explicit(&vert2edge_cnt[e0], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&vert2edge_cnt[e1], 1, memory_order_relaxed);
}

// Vertex-edge adjacency: fill
kernel void get_vertex_edge_adjacency_kernel(
    device const ulong* edges [[buffer(0)]],
    constant int& E [[buffer(1)]],
    device int* vert2edge [[buffer(2)]],
    device const int* vert2edge_offset [[buffer(3)]],
    device atomic_int* vert2edge_cnt [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    int idx0 = atomic_fetch_add_explicit(&vert2edge_cnt[e0], 1, memory_order_relaxed);
    vert2edge[vert2edge_offset[e0] + idx0] = (int)tid;
    int idx1 = atomic_fetch_add_explicit(&vert2edge_cnt[e1], 1, memory_order_relaxed);
    vert2edge[vert2edge_offset[e1] + idx1] = (int)tid;
}

// Set boundary vertex indicator
kernel void set_boundary_vertex_kernel(
    device const ulong* edges [[buffer(0)]],
    device const int* boundaries [[buffer(1)]],
    device const int* edge2face_cnt [[buffer(2)]],
    constant int& B [[buffer(3)]],
    device uint8_t* vert_is_boundary [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)B) return;
    int eid = boundaries[tid];
    if (edge2face_cnt[eid] == 1) {
        ulong e = edges[eid];
        int e0 = int(e >> 32);
        int e1 = int(e & 0xFFFFFFFF);
        vert_is_boundary[e0] = 1;
        vert_is_boundary[e1] = 1;
    }
}

// Vertex boundary adjacency: count
kernel void get_vertex_boundary_cnt_kernel(
    device const ulong* edges [[buffer(0)]],
    device const int* boundaries [[buffer(1)]],
    constant int& B [[buffer(2)]],
    device atomic_int* vert2bound_cnt [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)B) return;
    int eid = boundaries[tid];
    ulong e = edges[eid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    atomic_fetch_add_explicit(&vert2bound_cnt[e0], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&vert2bound_cnt[e1], 1, memory_order_relaxed);
}

// Vertex boundary adjacency: fill
kernel void get_vertex_boundary_adjacency_kernel(
    device const ulong* edges [[buffer(0)]],
    device const int* boundaries [[buffer(1)]],
    constant int& B [[buffer(2)]],
    device int* vert2bound [[buffer(3)]],
    device const int* vert2bound_offset [[buffer(4)]],
    device atomic_int* vert2bound_cnt [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)B) return;
    int eid = boundaries[tid];
    ulong e = edges[eid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    int idx0 = atomic_fetch_add_explicit(&vert2bound_cnt[e0], 1, memory_order_relaxed);
    vert2bound[vert2bound_offset[e0] + idx0] = (int)tid;
    int idx1 = atomic_fetch_add_explicit(&vert2bound_cnt[e1], 1, memory_order_relaxed);
    vert2bound[vert2bound_offset[e1] + idx1] = (int)tid;
}

// Vertex manifold check
kernel void get_vertex_is_manifold_kernel(
    device const int* vert2edge [[buffer(0)]],
    device const int* vert2edge_offset [[buffer(1)]],
    device const int* edge2face_cnt [[buffer(2)]],
    constant int& V [[buffer(3)]],
    device uint8_t* vert_is_manifold [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)V) return;
    int num_boundaries = 0;
    bool is_manifold = true;
    for (int i = vert2edge_offset[tid]; i < vert2edge_offset[tid+1]; i++) {
        int eid = vert2edge[i];
        if (edge2face_cnt[eid] == 1) {
            num_boundaries++;
            if (num_boundaries > 2) { is_manifold = false; break; }
        } else if (edge2face_cnt[eid] > 2) {
            is_manifold = false; break;
        }
    }
    vert_is_manifold[tid] = is_manifold ? 1 : 0;
}

// Set manifold face adjacency
kernel void set_manifold_face_adj_kernel(
    device const int* manifold_edge_idx [[buffer(0)]],
    device const int* edge2face [[buffer(1)]],
    device const int* edge2face_offset [[buffer(2)]],
    constant int& M [[buffer(3)]],
    device int2* manifold_face_adj [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;
    int edge_idx = manifold_edge_idx[tid];
    int start = edge2face_offset[edge_idx];
    int end = edge2face_offset[edge_idx + 1];
    if (end - start != 2) return;
    manifold_face_adj[tid] = int2(edge2face[start], edge2face[start + 1]);
}

// Set manifold boundary adjacency
kernel void set_manifold_bound_adj_kernel(
    device const int* manifold_boundary_verts_idx [[buffer(0)]],
    device const int* vert2bound [[buffer(1)]],
    device const int* vert2bound_offset [[buffer(2)]],
    constant int& MBV [[buffer(3)]],
    device int2* manifold_bound_adj [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)MBV) return;
    int vert_idx = manifold_boundary_verts_idx[tid];
    int b0 = vert2bound[vert2bound_offset[vert_idx]];
    int b1 = vert2bound[vert2bound_offset[vert_idx] + 1];
    manifold_bound_adj[tid] = int2(b0, b1);
}

// Check if boundary connected component is a loop
kernel void is_bound_conn_comp_loop_kernel(
    device const ulong* edges [[buffer(0)]],
    device const int* boundaries [[buffer(1)]],
    device const int* bound_conn_comp_ids [[buffer(2)]],
    device const int* vert2bound [[buffer(3)]],
    device const int* vert2bound_offset [[buffer(4)]],
    constant int& B [[buffer(5)]],
    device atomic_int* is_bound_conn_comp_loop [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)B) return;

    int eid = boundaries[tid];
    ulong e = edges[eid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    int self_comp_id = bound_conn_comp_ids[tid];

    int cnt = 0;
    for (int i = vert2bound_offset[e0]; i < vert2bound_offset[e0+1]; i++) {
        int b = vert2bound[i];
        if (b == (int)tid) continue;
        if (bound_conn_comp_ids[b] == self_comp_id) cnt++;
    }
    if (cnt == 0) {
        atomic_store_explicit(&is_bound_conn_comp_loop[self_comp_id], 0, memory_order_relaxed);
        return;
    }
    cnt = 0;
    for (int i = vert2bound_offset[e1]; i < vert2bound_offset[e1+1]; i++) {
        int b = vert2bound[i];
        if (b == (int)tid) continue;
        if (bound_conn_comp_ids[b] == self_comp_id) cnt++;
    }
    if (cnt == 0) {
        atomic_store_explicit(&is_bound_conn_comp_loop[self_comp_id], 0, memory_order_relaxed);
    }
}
