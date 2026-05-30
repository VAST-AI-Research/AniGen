// Atlas/chart clustering kernels
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

inline ulong pack_kv(int key, float value) {
    uint v = as_type<uint>(value);
    return (ulong(v) << 32) | ulong(uint(key));
}

// Initialize chart adjacency from face adjacency
kernel void init_chart_adj_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    device const int2* face_adj [[buffer(2)]],
    device const int* chart_ids [[buffer(3)]],
    constant int& M [[buffer(4)]],
    device ulong* chart_adj [[buffer(5)]],
    device float* length [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;
    int f0 = face_adj[tid].x, f1 = face_adj[tid].y;
    int c0 = chart_ids[f0], c1 = chart_ids[f1];
    if (c0 == c1) { chart_adj[tid] = ULONG_MAX; length[tid] = 0; return; }
    int mn = min(c0, c1), mx = max(c0, c1);
    chart_adj[tid] = (ulong(mn) << 32) | ulong(mx);

    int3 t0 = to_int3(faces[f0]), t1 = to_int3(faces[f1]);
    int t0v[3] = {t0.x, t0.y, t0.z};
    int common[2]; int found = 0;
    for (int i = 0; i < 3 && found < 2; i++) {
        int v = t0v[i];
        if (v == t1.x || v == t1.y || v == t1.z) { common[found++] = v; }
    }
    if (found >= 2) {
        float3 p0 = float3(vertices[common[0]]);
        float3 p1 = float3(vertices[common[1]]);
        float3 d = p0 - p1;
        length[tid] = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    } else { length[tid] = 0; }
}

// Count chart-edge connectivity + accumulate perimeters
// Uses atomic float add (Metal 3.0, Apple Silicon M1+)
kernel void get_chart_edge_cnt_kernel(
    device const ulong* chart_adj [[buffer(0)]],
    device const float* chart_adj_length [[buffer(1)]],
    constant int& E [[buffer(2)]],
    device atomic_int* chart2edge_cnt [[buffer(3)]],
    device atomic_float* chart_perim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong c = chart_adj[tid];
    float l = chart_adj_length[tid];
    int c0 = int(c >> 32), c1 = int(c & 0xFFFFFFFF);
    atomic_fetch_add_explicit(&chart2edge_cnt[c0], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&chart2edge_cnt[c1], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&chart_perim[c0], l, memory_order_relaxed);
    atomic_fetch_add_explicit(&chart_perim[c1], l, memory_order_relaxed);
}

// Fill chart-edge adjacency CSR
kernel void get_chart_edge_adjacency_kernel(
    device const ulong* chart_adj [[buffer(0)]],
    constant int& E [[buffer(1)]],
    device int* chart2edge [[buffer(2)]],
    device const int* chart2edge_offset [[buffer(3)]],
    device atomic_int* chart2edge_cnt [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong c = chart_adj[tid];
    int c0 = int(c >> 32), c1 = int(c & 0xFFFFFFFF);
    int idx0 = atomic_fetch_add_explicit(&chart2edge_cnt[c0], 1, memory_order_relaxed);
    chart2edge[chart2edge_offset[c0] + idx0] = (int)tid;
    int idx1 = atomic_fetch_add_explicit(&chart2edge_cnt[c1], 1, memory_order_relaxed);
    chart2edge[chart2edge_offset[c1] + idx1] = (int)tid;
}

// Compute chart adjacency cost (normal cone + area + perim/area penalties)
kernel void compute_chart_adjacency_cost_kernel(
    device const ulong* chart_adj [[buffer(0)]],
    device const float4* chart_normal_cones [[buffer(1)]],
    device const float* chart_adj_length [[buffer(2)]],
    device const float* chart_perims [[buffer(3)]],
    device const float* chart_areas [[buffer(4)]],
    constant float& area_penalty_weight [[buffer(5)]],
    constant float& perimeter_area_ratio_weight [[buffer(6)]],
    constant int& E [[buffer(7)]],
    device float* chart_adj_costs [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    ulong adj = chart_adj[tid];
    int c0 = int(adj >> 32), c1 = int(adj & 0xFFFFFFFF);

    float4 cone0 = chart_normal_cones[c0], cone1 = chart_normal_cones[c1];
    Vec3f a0 = Vec3f(cone0.xyz);
    Vec3f a1 = Vec3f(cone1.xyz);
    float ha0 = cone0.w, ha1 = cone1.w;
    float ca = a0.dot(a1);
    float aa = acos(clamp(ca, -1.0f, 1.0f));
    float ncl = min(-ha0, aa - ha1);
    float nch = max(ha0, aa + ha1);
    float nha = (nch - ncl) * 0.5f;
    float cost = nha;

    float new_area = chart_areas[c0] + chart_areas[c1];
    cost += area_penalty_weight * new_area;
    float new_perim = chart_perims[c0] + chart_perims[c1] - 2 * chart_adj_length[tid];
    cost += perimeter_area_ratio_weight * (new_perim * new_perim / new_area);
    chart_adj_costs[tid] = cost;
}

// Propagate minimum cost to each chart
kernel void chart_propagate_cost_kernel(
    device const int* chart2edge [[buffer(0)]],
    device const int* chart2edge_offset [[buffer(1)]],
    device const float* edge_collapse_costs [[buffer(2)]],
    constant int& num_charts [[buffer(3)]],
    device ulong* propagated_costs [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)num_charts) return;
    int min_eid = -1;
    float min_cost = INFINITY;
    for (int e = chart2edge_offset[tid]; e < chart2edge_offset[tid+1]; e++) {
        int eid = chart2edge[e];
        float cost = edge_collapse_costs[eid];
        if (cost < min_cost || (cost == min_cost && eid < min_eid)) {
            min_eid = eid; min_cost = cost;
        }
    }
    propagated_costs[tid] = pack_kv(min_eid, min_cost);
}

// Collapse chart edges
kernel void chart_collapse_edges_kernel(
    device ulong* chart_adj [[buffer(0)]],
    device const float* edge_collapse_costs [[buffer(1)]],
    device const ulong* propagated_costs [[buffer(2)]],
    constant float& collapse_thresh [[buffer(3)]],
    constant int& E [[buffer(4)]],
    device int* chart_map [[buffer(5)]],
    device float4* chart_normal_cones [[buffer(6)]],
    device atomic_int* end_flag [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)E) return;
    float cost = edge_collapse_costs[tid];
    if (cost > collapse_thresh) return;

    ulong c = chart_adj[tid];
    int c0 = int(c >> 32), c1 = int(c & 0xFFFFFFFF);
    ulong pack = pack_kv((int)tid, cost);
    if (propagated_costs[c0] < pack || propagated_costs[c1] < pack) return;

    chart_map[c1] = c0;

    // Update cone
    float4 cn0 = chart_normal_cones[c0], cn1 = chart_normal_cones[c1];
    Vec3f a0 = Vec3f(cn0.xyz);
    Vec3f a1 = Vec3f(cn1.xyz);
    float ha0 = cn0.w, ha1 = cn1.w;
    float ca = a0.dot(a1);
    float aa = acos(clamp(ca, -1.0f, 1.0f));
    float ncl = min(-ha0, aa - ha1);
    float nch = max(ha0, aa + ha1);
    float nha = (nch - ncl) * 0.5f;
    Vec3f new_axis;
    if (aa < 1e-3f) { new_axis = a0; }
    else {
        float naa = (nch + ncl) * 0.5f;
        new_axis = a0 * cos(naa) + (a1 - a0 * ca).normalized() * sin(naa);
        new_axis.normalize();
    }
    chart_normal_cones[c0] = float4(new_axis.to_float3(), nha);
    atomic_store_explicit(end_flag, 0, memory_order_relaxed);
}

// Refine chart assignment per face
kernel void refine_charts_kernel(
    device const float4* chart_normal_cones [[buffer(0)]],
    device const packed_float3* face_normals [[buffer(1)]],
    device const packed_float3* vertices [[buffer(2)]],
    device const ulong* edges [[buffer(3)]],
    device const packed_int3* face2edge [[buffer(4)]],
    device const int* edge2face [[buffer(5)]],
    device const int* edge2face_offset [[buffer(6)]],
    constant int& F [[buffer(7)]],
    constant float& lambda_smooth [[buffer(8)]],
    device const int* chart_ids [[buffer(9)]],
    device int* pong_chart_ids [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int current_c = chart_ids[tid];
    Vec3f n = Vec3f(float3(face_normals[tid]));

    int candidates[4]; float smooth[4]; int nc = 0;
    candidates[0] = current_c; smooth[0] = 0; nc = 1;

    int eids[3] = {face2edge[tid].x, face2edge[tid].y, face2edge[tid].z};
    for (int i = 0; i < 3; i++) {
        int eid = eids[i];
        int v0i = int(edges[eid] >> 32), v1i = int(edges[eid] & 0xFFFFFFFF);
        Vec3f v0 = Vec3f(float3(vertices[v0i]));
        Vec3f v1 = Vec3f(float3(vertices[v1i]));
        float el = (v1 - v0).norm();
        for (int j = edge2face_offset[eid]; j < edge2face_offset[eid+1]; j++) {
            int nf = edge2face[j];
            if (nf == (int)tid) continue;
            int nc_id = chart_ids[nf];
            int idx = -1;
            for (int k = 0; k < nc; k++) { if (candidates[k] == nc_id) { idx = k; break; } }
            if (idx == -1 && nc < 4) { idx = nc++; candidates[idx] = nc_id; smooth[idx] = 0; }
            if (idx != -1) smooth[idx] += el;
        }
    }

    int best_c = current_c; float best = -1e9f;
    for (int i = 0; i < nc; i++) {
        float4 cone = chart_normal_cones[candidates[i]];
        Vec3f axis = Vec3f(cone.xyz);
        float geo = axis.dot(n);
        if (geo <= 0) continue;
        float total = geo + smooth[i] * lambda_smooth;
        if (candidates[i] == current_c && best == -1e9f) { best = total; best_c = candidates[i]; }
        float diff = total - best;
        if (diff > 1e-5f) { best = total; best_c = candidates[i]; }
        else if (abs(diff) <= 1e-5f && candidates[i] < best_c) { best = total; best_c = candidates[i]; }
    }
    pong_chart_ids[tid] = best_c;
}

// Hook edges only if same chart (for reassignment)
kernel void hook_edges_if_same_chart_kernel(
    device const int2* adj [[buffer(0)]],
    device const int* chart_ids [[buffer(1)]],
    constant int& M [[buffer(2)]],
    device atomic_int* conn_comp_ids [[buffer(3)]],
    device atomic_int* end_flag [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)M) return;
    int f0 = adj[tid].x, f1 = adj[tid].y;
    if (chart_ids[f0] != chart_ids[f1]) return;

    int root0 = atomic_load_explicit(&conn_comp_ids[f0], memory_order_relaxed);
    while (root0 != atomic_load_explicit(&conn_comp_ids[root0], memory_order_relaxed))
        root0 = atomic_load_explicit(&conn_comp_ids[root0], memory_order_relaxed);
    int root1 = atomic_load_explicit(&conn_comp_ids[f1], memory_order_relaxed);
    while (root1 != atomic_load_explicit(&conn_comp_ids[root1], memory_order_relaxed))
        root1 = atomic_load_explicit(&conn_comp_ids[root1], memory_order_relaxed);
    if (root0 == root1) return;
    int high = max(root0, root1), low = min(root0, root1);
    atomic_fetch_min_explicit(&conn_comp_ids[high], low, memory_order_relaxed);
    atomic_store_explicit(end_flag, 0, memory_order_relaxed);
}

// Normal difference per face vs chart average normal
kernel void normal_diff_kernel(
    device const packed_float3* chart_normals [[buffer(0)]],
    device const packed_float3* sorted_face_normals [[buffer(1)]],
    device const int* sorted_chart_ids [[buffer(2)]],
    constant int& F [[buffer(3)]],
    device float* normal_diff [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int c = sorted_chart_ids[tid];
    Vec3f n = Vec3f(float3(chart_normals[c]));
    Vec3f fn = Vec3f(float3(sorted_face_normals[tid]));
    normal_diff[tid] = acos(clamp(n.dot(fn), -1.0f, 1.0f));
}

// Normalize float3 array
kernel void normalize_float3_kernel(
    device packed_float3* data [[buffer(0)]],
    constant int& N [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    float3 v = float3(data[tid]);
    float n = length(v);
    if (n > 0) data[tid] = packed_float3(v / n);
}

// Update chart normal cones from normals + half angles
kernel void update_normal_cones_kernel(
    device float4* cones [[buffer(0)]],
    device const packed_float3* normals [[buffer(1)]],
    device const float* half_angles [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    float3 n = float3(normals[tid]);
    cones[tid] = float4(n, half_angles[tid]);
}

// Expand chart_ids+vertex_ids into packed uint64 for chart mesh construction
kernel void expand_chart_ids_and_vertex_ids_kernel(
    device const int* sorted_chart_ids [[buffer(0)]],
    device const int* sorted_face_idx [[buffer(1)]],
    device const packed_int3* faces [[buffer(2)]],
    constant int& F [[buffer(3)]],
    device ulong* pack [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int c = sorted_chart_ids[tid];
    int3 face = to_int3(faces[sorted_face_idx[tid]]);
    pack[3*tid+0] = (ulong(c) << 32) | ulong(face.x);
    pack[3*tid+1] = (ulong(c) << 32) | ulong(face.y);
    pack[3*tid+2] = (ulong(c) << 32) | ulong(face.z);
}

// Unpack faces from compressed IDs
kernel void unpack_faces_kernel(
    device const ulong* pack [[buffer(0)]],
    constant int& F [[buffer(1)]],
    device packed_int3* faces [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    faces[tid] = from_int3(int3(int(pack[3*tid]), int(pack[3*tid+1]), int(pack[3*tid+2])));
}

// Unpack vertex IDs and offsets from compressed IDs
kernel void unpack_vertex_ids_kernel(
    device const ulong* pack [[buffer(0)]],
    constant int& N [[buffer(1)]],
    device int* vertex_ids [[buffer(2)]],
    device int* vertex_offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    vertex_ids[tid] = int(pack[tid] & 0xFFFFFFFF);
    int cur_c = int(pack[tid] >> 32);
    if (tid == 0) vertex_offsets[0] = 0;
    else {
        int prev_c = int(pack[tid-1] >> 32);
        if (cur_c != prev_c) vertex_offsets[cur_c] = (int)tid;
    }
    if (tid == (uint)N - 1) vertex_offsets[cur_c + 1] = N;
}
