#include "cumesh.h"
#include "dtypes.cuh"
#include "shared.h"
#include <cub/cub.cuh>


namespace cumesh {


/*
Fast mesh parameterization / UV unwrapping using GPU

Three main steps:
1. Split the mesh into charts
   - Treat each chart as a node in a graph
   - Use a parallel edge collapse algorithm to merge charts based on normal cone deviation
2. Parameterize each chart using Least Squares Conformal Maps (LSCM)
3. Pack the charts into a texture atlas
*/


__device__ inline uint64_t pack_key_value_positive(int key, float value) {
    unsigned int v = __float_as_uint(value);
    return (static_cast<uint64_t>(v) << 32) |
           static_cast<unsigned int>(key);
}


__device__ inline void unpack_key_value_positive(uint64_t key_value, int& key, float& value) {
    key = static_cast<int>(key_value & 0xffffffffu);
    value = __uint_as_float(static_cast<unsigned int>(key_value >> 32));
}


// static __global__ void init_normal_cones_kernel(
//     const float3* face_normals,
//     const int F,
//     float4* chart_normal_cones
// ) {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= F) return;
// 
//     float3 n = face_normals[tid];
//     chart_normal_cones[tid] = make_float4(n.x, n.y, n.z, 0.0f); // half angle = 0
// }


static __global__ void init_chart_adj_kernel(
    const float3* vertices,
    const int3* faces,
    const int2* face_adj,
    const int* chart_ids,
    const size_t M,
    uint64_t* chart_adj,
    float* length
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    int f0 = face_adj[tid].x;
    int f1 = face_adj[tid].y;
    
    int c0 = chart_ids[f0];
    int c1 = chart_ids[f1];

    if (c0 == c1) {
        chart_adj[tid] = UINT64_MAX;
        length[tid] = 0.0f;
        return;
    }

    int min_c = min(c0, c1);
    int max_c = max(c0, c1);
    chart_adj[tid] = (static_cast<uint64_t>(min_c) << 32) | static_cast<uint64_t>(max_c);

    int3 tri0 = faces[f0];
    int3 tri1 = faces[f1];

    int t0_indices[3] = {tri0.x, tri0.y, tri0.z};
    int common_v_indices[2]; 
    int found_count = 0;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        int v = t0_indices[i];
        if (v == tri1.x || v == tri1.y || v == tri1.z) {
            if (found_count < 2) {
                common_v_indices[found_count] = v;
            }
            found_count++;
        }
    }

    if (found_count >= 2) {
        float3 p0 = vertices[common_v_indices[0]];
        float3 p1 = vertices[common_v_indices[1]];

        float dx = p0.x - p1.x;
        float dy = p0.y - p1.y;
        float dz = p0.z - p1.z;

        length[tid] = sqrtf(dx * dx + dy * dy + dz * dz);
    } else {
        length[tid] = 0.0f;
    }
}


static __global__ void get_chart_edge_cnt_kernel(
    const uint64_t* chart_adj,
    const float* chart_adj_length,
    const int E,
    int* chart2edge_cnt,
    float* chart_perim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t c = chart_adj[tid];
    float l = chart_adj_length[tid];
    int c0 = int(c >> 32);
    int c1 = int(c & 0xFFFFFFFF);

    // count vertex adjacent edge number
    atomicAdd(&chart2edge_cnt[c0], 1);
    atomicAdd(&chart2edge_cnt[c1], 1);
    atomicAdd(&chart_perim[c0], l);
    atomicAdd(&chart_perim[c1], l);
}


static __global__ void get_chart_edge_adjacency_kernel(
    const uint64_t* chart_adj,
    const int E,
    int* chart2edge,
    int* chart2edge_offset,
    int* chart2edge_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t c = chart_adj[tid];
    int c0 = int(c >> 32);
    int c1 = int(c & 0xFFFFFFFF);

    // assign connectivity
    chart2edge[chart2edge_offset[c0] + atomicAdd(&chart2edge_cnt[c0], 1)] = tid;
    chart2edge[chart2edge_offset[c1] + atomicAdd(&chart2edge_cnt[c1], 1)] = tid;
}


static __global__ void compute_chart_adjacency_cost_kernel(
    const uint64_t* chart_adj,
    const float4* chart_normal_cones,
    const float* chart_adj_length,
    const float* chart_perims,
    const float* chart_areas,
    float area_penalty_weight,
    float perimeter_area_ratio_weight,
    const int E,
    float* chart_adj_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    uint64_t adj = chart_adj[tid];
    int c0 = static_cast<int>(adj >> 32);
    int c1 = static_cast<int>(adj & 0xFFFFFFFF);

    float4 cone0 = chart_normal_cones[c0];
    float4 cone1 = chart_normal_cones[c1];
    Vec3f axis0(cone0.x, cone0.y, cone0.z);
    Vec3f axis1(cone1.x, cone1.y, cone1.z);
    float half_angle0 = cone0.w;
    float half_angle1 = cone1.w;
    float cos_angle = axis0.dot(axis1);
    float axis_angle = acosf(fmaxf(fminf(cos_angle, 1.0f), -1.0f));
    float new_cone_low = fminf(-half_angle0, axis_angle - half_angle1);
    float new_cone_high = fmaxf(half_angle0, axis_angle + half_angle1);
    float new_half_angle = (new_cone_high - new_cone_low) * 0.5f;
    float cost = new_half_angle;

    // Chart area panelty
    float new_area = (chart_areas[c0] + chart_areas[c1]);
    cost += area_penalty_weight * new_area;

    // Perim-area ration panelty
    float new_perim = chart_perims[c0] + chart_perims[c1] - 2 * chart_adj_length[tid];
    cost += perimeter_area_ratio_weight * (new_perim * new_perim / new_area);

    chart_adj_costs[tid] = cost;
}


static __global__ void propagate_cost_kernel(
    const int* chart2edge,
    const int* chart2edge_offset,
    const float* edge_collapse_costs,
    const int num_charts,
    uint64_t* propagated_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_charts) return;

    // get edge with minimum cost
    int min_eid = -1;
    float min_cost = FLT_MAX;
    for (int e = chart2edge_offset[tid]; e < chart2edge_offset[tid+1]; e++) {
        int eid = chart2edge[e];
        float cost = edge_collapse_costs[eid];
        if (cost < min_cost || (cost == min_cost && eid < min_eid)) {
            min_eid = eid;
            min_cost = cost;
        }
    }

    uint64_t cost = pack_key_value_positive(min_eid, min_cost);
    propagated_costs[tid] = cost;
}


static __global__ void collapse_edges_kernel(
    uint64_t* chart_adj,
    const float* edge_collapse_costs,
    const uint64_t* propagated_costs,
    const float collapse_thresh,
    const int E,
    int* chart_map,
    float4* chart_normal_cones,
    int* end_flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    float cost = edge_collapse_costs[tid];
    if (cost > collapse_thresh) return;

    // get edge
    uint64_t c = chart_adj[tid];
    int c0 = int(c >> 32);
    int c1 = int(c & 0xFFFFFFFF);

    // check if this edge has the minimum cost among neighboring charts
    uint64_t pack = pack_key_value_positive(tid, cost);
    if (propagated_costs[c0] < pack || propagated_costs[c1] < pack) return;

    // collapse edge
    chart_map[c1] = c0;

    // update cone
    float4 cone0 = chart_normal_cones[c0];
    float4 cone1 = chart_normal_cones[c1];
    Vec3f axis0(cone0.x, cone0.y, cone0.z);
    Vec3f axis1(cone1.x, cone1.y, cone1.z);
    float half_angle0 = cone0.w;
    float half_angle1 = cone1.w;
    float cos_angle = axis0.dot(axis1);
    float axis_angle = acosf(fmaxf(fminf(cos_angle, 1.0f), -1.0f));
    float new_cone_low = fminf(-half_angle0, axis_angle - half_angle1);
    float new_cone_high = fmaxf(half_angle0, axis_angle + half_angle1);
    float new_half_angle = (new_cone_high - new_cone_low) * 0.5f;
    Vec3f new_axis;
    if (axis_angle < 1e-3f) {
        new_axis = axis0;
    } else {
        float new_axis_angle = (new_cone_high + new_cone_low) * 0.5f;
        new_axis = axis0 * cosf(new_axis_angle) + (axis1 - axis0 * cos_angle).normalized() * sinf(new_axis_angle);
        new_axis.normalize();
    }
    chart_normal_cones[c0] = make_float4(new_axis.x, new_axis.y, new_axis.z, new_half_angle);

    // not end of iteration
    *end_flag = 0;
}


static void get_chart_connectivity(
    CuMesh& mesh
) {
    size_t M = mesh.manifold_face_adj.size;

    // 1. Get chart adjacency
    // 1.1 Initialize chart adjacency and edge lengths
    mesh.atlas_chart_adj.resize(M);
    mesh.atlas_chart_adj_length.resize(M);
    float *cu_raw_lengths, *cu_sorted_lengths;
    CUDA_CHECK(cudaMalloc(&cu_raw_lengths, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cu_sorted_lengths, M * sizeof(float)));

    init_chart_adj_kernel<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.vertices.ptr,
        mesh.faces.ptr,
        mesh.manifold_face_adj.ptr,
        mesh.atlas_chart_ids.ptr,
        M,
        mesh.atlas_chart_adj.ptr,
        cu_raw_lengths
    );
    CUDA_CHECK(cudaGetLastError());

    // 1.2 Sort
    size_t temp_storage_bytes = 0;
    mesh.temp_storage.resize(M * sizeof(uint64_t));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        mesh.atlas_chart_adj.ptr,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        cu_raw_lengths,
        cu_sorted_lengths,
        M
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        mesh.atlas_chart_adj.ptr,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        cu_raw_lengths,
        cu_sorted_lengths,
        M
    ));
    CUDA_CHECK(cudaFree(cu_raw_lengths));
	
    #if CUDART_VERSION >= 12090
        auto reduce_op = ::cuda::std::plus();
    #else
        auto reduce_op = cub::Sum();
    #endif
	

    // 1.3 Reduce By Key (Aggregate duplicate chart pairs by summing lengths)
    int* cu_num_chart_adjs;
    CUDA_CHECK(cudaMalloc(&cu_num_chart_adjs, sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
		reduce_op,
        M
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
		reduce_op,
        M
    ));
    CUDA_CHECK(cudaMemcpy(&mesh.atlas_chart_adj.size, cu_num_chart_adjs, sizeof(int), cudaMemcpyDeviceToHost));
    mesh.atlas_chart_adj_length.size = mesh.atlas_chart_adj.size;
    CUDA_CHECK(cudaFree(cu_sorted_lengths));
    CUDA_CHECK(cudaFree(cu_num_chart_adjs));
    // Remove invalid edge (UINT64_MAX) if present
    // Since we sorted, invalid edges are at the end.
    uint64_t last_key;
    if (mesh.atlas_chart_adj.size > 0) {
        CUDA_CHECK(cudaMemcpy(&last_key, mesh.atlas_chart_adj.ptr + mesh.atlas_chart_adj.size - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (last_key == UINT64_MAX) { 
            mesh.atlas_chart_adj.size -= 1;
            mesh.atlas_chart_adj_length.size -= 1;
        }
    }
    // Early stop if no valid edges
    if (mesh.atlas_chart_adj.size == 0) {
        return;
    }

    // 2. Get chart-edge connectivity
    size_t E = mesh.atlas_chart_adj.size;
    size_t C = mesh.atlas_num_charts;
    // 2.1 Count edge number for each chart, along with perim
    mesh.atlas_chart2edge_cnt.resize(C);
    mesh.atlas_chart2edge_cnt.zero();
    mesh.atlas_chart_perims.resize(C);
    mesh.atlas_chart_perims.zero();
    get_chart_edge_cnt_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.atlas_chart_adj.ptr,
        mesh.atlas_chart_adj_length.ptr,
        E,
        mesh.atlas_chart2edge_cnt.ptr,
        mesh.atlas_chart_perims.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    // 2.2 Prepare CSR format for chart-edge connectivity
    mesh.atlas_chart2edge_offset.resize(C + 1);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        mesh.atlas_chart2edge_cnt.ptr,
        mesh.atlas_chart2edge_offset.ptr,
        C + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        mesh.atlas_chart2edge_cnt.ptr,
        mesh.atlas_chart2edge_offset.ptr,
        C + 1
    ));
    // 2.3 Fill CSR format for chart-edge connectivity
    mesh.atlas_chart2edge.resize(2 * E); // each edge connects two charts
    mesh.atlas_chart2edge_cnt.zero();
    get_chart_edge_adjacency_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.atlas_chart_adj.ptr,
        E,
        mesh.atlas_chart2edge.ptr,
        mesh.atlas_chart2edge_offset.ptr,
        mesh.atlas_chart2edge_cnt.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


struct Float3Add
{
    __host__ __device__
    float3 operator()(const float3 &a, const float3 &b) const
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};


static __global__ void normalize_kernel(
    float3* chart_normals,
    const int num_charts
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_charts) return;

    float3 n = chart_normals[tid];
    float norm = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (norm > 0.0f) {
        n.x /= norm;
        n.y /= norm;
        n.z /= norm;
    }
    chart_normals[tid] = n;
}


static __global__ void normal_diff_kernel(
    const float3* chart_normals,
    const float3* sorted_face_normals,
    const int* sorted_chart_ids,
    const size_t F,
    float* normal_diff
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int c = sorted_chart_ids[tid];
    Vec3f n(chart_normals[c]);
    Vec3f fn(sorted_face_normals[tid]);
    normal_diff[tid] = acosf(fmaxf(fminf(n.dot(fn), 1.0f), -1.0f));
}


static __global__ void update_normal_cones_kernel(
    float4* chart_normal_cones,
    const float3* chart_normals,
    const float* new_cone_half_angles,
    const int num_charts
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_charts) return;

    float3 n = chart_normals[tid];
    float half_angle = new_cone_half_angles[tid];
    chart_normal_cones[tid] = make_float4(n.x, n.y, n.z, half_angle);
}


void compute_chart_normal_cones(
    CuMesh& mesh
) {
    size_t C = mesh.atlas_num_charts;
    size_t F = mesh.faces.size;

    // 1. Sort faces by chart id
    int* sorted_chart_ids;
    int* faces_ids;
    int* argsorted_faces_ids;
    CUDA_CHECK(cudaMalloc(&sorted_chart_ids, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&faces_ids, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&argsorted_faces_ids, F * sizeof(int)));
    arange_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        faces_ids,
        F
    );
    CUDA_CHECK(cudaGetLastError());
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        mesh.atlas_chart_ids.ptr, sorted_chart_ids,
        faces_ids, argsorted_faces_ids,
        F
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        mesh.atlas_chart_ids.ptr, sorted_chart_ids,
        faces_ids, argsorted_faces_ids,
        F
    ));
    CUDA_CHECK(cudaFree(faces_ids));
    
    // 2. Get CSR format for chart-face assignment
    int* cu_chart_size;
    int* cu_num_charts;
    int* cu_unique_chart_ids;
    CUDA_CHECK(cudaMalloc(&cu_chart_size, (C + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_num_charts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_unique_chart_ids, (C + 1) * sizeof(int)));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_bytes,
        sorted_chart_ids, cu_unique_chart_ids, cu_chart_size, cu_num_charts,
        F
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        sorted_chart_ids, cu_unique_chart_ids, cu_chart_size, cu_num_charts,
        F
    ));
    CUDA_CHECK(cudaFree(cu_num_charts));
    CUDA_CHECK(cudaFree(cu_unique_chart_ids));

    int* cu_chart_offsets;
    CUDA_CHECK(cudaMalloc(&cu_chart_offsets, (C + 1) * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_chart_size, cu_chart_offsets,
        C + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        cu_chart_size, cu_chart_offsets,
        C + 1
    ));
    CUDA_CHECK(cudaFree(cu_chart_size));

    // 3. Compute chart normals and areas
    float* cu_sorted_face_areas;
    CUDA_CHECK(cudaMalloc(&cu_sorted_face_areas, F * sizeof(float)));
    index_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.face_areas.ptr,
        argsorted_faces_ids,
        F,
        cu_sorted_face_areas
    );
    CUDA_CHECK(cudaGetLastError());
    mesh.atlas_chart_areas.resize(C);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_sorted_face_areas, mesh.atlas_chart_areas.ptr,
        C,
        cu_chart_offsets, cu_chart_offsets + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_face_areas, mesh.atlas_chart_areas.ptr,
        C,
        cu_chart_offsets, cu_chart_offsets + 1
    ));
    CUDA_CHECK(cudaFree(cu_sorted_face_areas));

    float3* cu_sorted_face_normals;
    CUDA_CHECK(cudaMalloc(&cu_sorted_face_normals, F * sizeof(float3)));
    index_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.face_normals.ptr,
        argsorted_faces_ids,
        F,
        cu_sorted_face_normals
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(argsorted_faces_ids));
    float3* cu_chart_normals;
    CUDA_CHECK(cudaMalloc(&cu_chart_normals, C * sizeof(float3)));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_bytes,
        cu_sorted_face_normals, cu_chart_normals,
        C,
        cu_chart_offsets, cu_chart_offsets + 1,
        Float3Add(),
        make_float3(0.0f, 0.0f, 0.0f)
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_face_normals, cu_chart_normals,
        C,
        cu_chart_offsets, cu_chart_offsets + 1,
        Float3Add(),
        make_float3(0.0f, 0.0f, 0.0f)
    ));
    normalize_kernel<<<(C + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_chart_normals,
        C
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Compute normal difference
    float* cu_normal_diff;
    CUDA_CHECK(cudaMalloc(&cu_normal_diff, F * sizeof(float)));
    normal_diff_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_chart_normals,
        cu_sorted_face_normals,
        sorted_chart_ids,
        F,
        cu_normal_diff
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_sorted_face_normals));
    CUDA_CHECK(cudaFree(sorted_chart_ids));

    // 5. Compute new cone half angles
    float* cu_new_cone_half_angles;
    CUDA_CHECK(cudaMalloc(&cu_new_cone_half_angles, C * sizeof(float)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Max(
        nullptr, temp_storage_bytes,
        cu_normal_diff, cu_new_cone_half_angles,
        C,
        cu_chart_offsets, cu_chart_offsets + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Max(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        cu_normal_diff, cu_new_cone_half_angles,
        C,
        cu_chart_offsets, cu_chart_offsets + 1
    ));
    CUDA_CHECK(cudaFree(cu_chart_offsets));
    CUDA_CHECK(cudaFree(cu_normal_diff));

    // 6. Update chart normal cones
    mesh.atlas_chart_normal_cones.resize(C);
    update_normal_cones_kernel<<<(C + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.atlas_chart_normal_cones.ptr,
        cu_chart_normals,
        cu_new_cone_half_angles,
        C
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_chart_normals));
    CUDA_CHECK(cudaFree(cu_new_cone_half_angles));
}


static __global__ void refine_charts_kernel(
    const float4* chart_normal_cones,
    const float3* face_normals,
    const float3* vertices,
    const uint64_t* edges,
    const int3* face2edge,
    const int* edge2face,
    const int* edge2face_offset,
    const size_t F,
    const float lambda_smooth,
    const int* chart_ids,         // Read-only (Input)
    int* pong_chart_ids           // Write-only (Output)
) {
    const int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= F) return;

    // 1. Load current face data
    int current_c = chart_ids[fid];
    Vec3f n(face_normals[fid]); 

    // local register cache for candidate list (triangle has at most 3 neighbors, plus self, max 4 candidates)
    int candidates[4];
    float smooth_scores[4];
    int num_candidates = 0;

    // init: add self to candidate list
    candidates[0] = current_c;
    smooth_scores[0] = 0.0f;
    num_candidates = 1;

    // 2. Iterate over 3 edges to aggregate smooth scores
    int eids[3] = { face2edge[fid].x, face2edge[fid].y, face2edge[fid].z };
    
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        int eid = eids[i];

        // calculate edge length (as smooth weight)
        // logic: if I add the neighbor's Chart, I can eliminate this edge as a boundary cost
        int v0_idx = int(edges[eid] >> 32);
        int v1_idx = int(edges[eid] & 0xFFFFFFFF);
        Vec3f v0 = Vec3f(vertices[v0_idx]);
        Vec3f v1 = Vec3f(vertices[v1_idx]);
        float edge_len = (v1 - v0).norm();

        int start = edge2face_offset[eid];
        int end = edge2face_offset[eid + 1];

        // Process edge neighbors
        for (int j = start; j < end; j++) {
            int neighbor_fid = edge2face[j];
            if (neighbor_fid == fid) continue;

            int neighbor_c = chart_ids[neighbor_fid]; // Read from Input buffer

            int idx = -1;
            for (int k = 0; k < num_candidates; ++k) {
                if (candidates[k] == neighbor_c) {
                    idx = k;
                    break;
                }
            }

            if (idx == -1 && num_candidates < 4) {
                idx = num_candidates++;
                candidates[idx] = neighbor_c;
                smooth_scores[idx] = 0.0f;
            }

            if (idx != -1) {
                smooth_scores[idx] += edge_len;
            }
        }
    }

    // 3. Evaluate candidates and pick best
    int best_c = current_c;
    float best_total_score = -1e9f;

    for (int i = 0; i < num_candidates; ++i) {
        int c = candidates[i];
        
        // A. Geom score
        float4 cone = chart_normal_cones[c];
        Vec3f axis(cone.x, cone.y, cone.z);
        float geo_sim = axis.dot(n); // [-1, 1]

        // if invalid cone, skip
        if (geo_sim <= 0.0f) continue;

        // B. Smooth score
        float smooth_sim = smooth_scores[i] * lambda_smooth;

        float total_score = geo_sim + smooth_sim;

        if (c == current_c) {
            if (best_total_score == -1e9f) {
                best_total_score = total_score;
                best_c = c;
            }
        }

        // C. Compare with best
        float diff = total_score - best_total_score;
        const float epsilon = 1e-5f; // dampening factor

        if (diff > epsilon) {
            // new best is significantly better than current best
            best_total_score = total_score;
            best_c = c;
        } 
        else if (abs(diff) <= epsilon) {
            // scores are very close, break tie by choosing smaller ID
            if (c < best_c) {
                best_total_score = total_score;
                best_c = c;
            }
        }
    }

    // Write back to Output buffer
    pong_chart_ids[fid] = best_c;
}


__global__ void hook_edges_if_same_chart_kernel(
    const int2* adj,
    const int* chart_ids,
    const int M,
    int* conn_comp_ids,
    int* end_flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    // get adjacent faces
    int f0 = adj[tid].x;
    int f1 = adj[tid].y;
    int c0 = chart_ids[f0];
    int c1 = chart_ids[f1];
    if (c0 != c1) return;

    // union
    // find roots
    int root0 = conn_comp_ids[f0];
    while (root0 != conn_comp_ids[root0]) {
        root0 = conn_comp_ids[root0];
    }
    int root1 = conn_comp_ids[f1];
    while (root1 != conn_comp_ids[root1]) {
        root1 = conn_comp_ids[root1];
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomicMin(&conn_comp_ids[high], low);
    *end_flag = 0;
}


static void reassign_chart_ids(
    CuMesh& mesh
) {
    size_t F = mesh.faces.size;
    size_t M = mesh.manifold_face_adj.size;

    mesh.temp_storage.resize(F * sizeof(int));        // Use as parent for DSU
    arange_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<int*>(mesh.temp_storage.ptr),
        F
    );
    CUDA_CHECK(cudaGetLastError());

    int* cu_end_flag; int h_end_flag;
    CUDA_CHECK(cudaMalloc(&cu_end_flag, sizeof(int)));
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag, &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_if_same_chart_kernel<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            mesh.manifold_face_adj.ptr,
            mesh.atlas_chart_ids.ptr,
            M,
            reinterpret_cast<int*>(mesh.temp_storage.ptr),
            cu_end_flag
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            reinterpret_cast<int*>(mesh.temp_storage.ptr),
            F
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);
    CUDA_CHECK(cudaFree(cu_end_flag));
    
    swap_buffers(mesh.atlas_chart_ids, mesh.temp_storage);
    mesh.atlas_num_charts = compress_ids(mesh.atlas_chart_ids.ptr, F, mesh.cub_temp_storage);
}


static __global__ void expand_chart_ids_and_vertex_ids_kernel(
    const int* sorted_chart_ids,
    const int* sorted_face_idx,
    const int3* faces,
    const size_t F,
    uint64_t* pack
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int c = sorted_chart_ids[tid];
    int f = sorted_face_idx[tid];
    int3 face = faces[f];
    int v0 = face.x;
    int v1 = face.y;
    int v2 = face.z;

    pack[3 * tid + 0] = (uint64_t(c) << 32) | v0;
    pack[3 * tid + 1] = (uint64_t(c) << 32) | v1;
    pack[3 * tid + 2] = (uint64_t(c) << 32) | v2;
}


static __global__ void unpack_faces_kernel(
    const uint64_t* pack,
    const size_t F,
    int3* faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    int3 face;
    face.x = int(pack[3 * tid + 0]);
    face.y = int(pack[3 * tid + 1]);
    face.z = int(pack[3 * tid + 2]);
    faces[tid] = face;
}


static __global__ void unpack_vertex_ids_kernel(
    const uint64_t* pack,
    const size_t N,
    int* vertex_ids,
    int* vertex_offsets
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    vertex_ids[tid] = int(pack[tid] & 0xFFFFFFFF);

    int cur_c = int(pack[tid] >> 32);
    if (tid == 0) {
        vertex_offsets[0] = 0;
    }
    else {
        int prev_c = int(pack[tid - 1] >> 32);
        if (cur_c != prev_c) {
            vertex_offsets[cur_c] = tid;
        }
    }
    if (tid == N - 1) {
        vertex_offsets[cur_c + 1] = N;
    }
}


void construct_chart_mesh(
    CuMesh& mesh
) {
    size_t F = mesh.faces.size;

    // 1. Sort faces by chart id
    mesh.atlas_chart_faces.resize(F);
    mesh.atlas_chart_faces_offset.resize(mesh.atlas_num_charts + 1);
    int* cu_sorted_chart_ids;
    int* cu_face_idx;
    int* cu_sorted_face_idx;
    CUDA_CHECK(cudaMalloc(&cu_sorted_chart_ids, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_face_idx, F * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_sorted_face_idx, F * sizeof(int)));
    arange_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_face_idx,
        F
    );
    CUDA_CHECK(cudaGetLastError());
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        mesh.atlas_chart_ids.ptr, cu_sorted_chart_ids,
        cu_face_idx, cu_sorted_face_idx,
        F
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        mesh.atlas_chart_ids.ptr, cu_sorted_chart_ids,
        cu_face_idx, cu_sorted_face_idx,
        F
    ));
    CUDA_CHECK(cudaFree(cu_face_idx));
    // 2. RLE for chart size
    int* cu_chart_size;
    int* cu_num_chart;
    int* cu_unique_chart_ids;
    CUDA_CHECK(cudaMalloc(&cu_chart_size, (mesh.atlas_num_charts + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_num_chart, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_unique_chart_ids, mesh.atlas_num_charts * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_bytes,
        cu_sorted_chart_ids, cu_unique_chart_ids, cu_chart_size, cu_num_chart,
        F
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_chart_ids, cu_unique_chart_ids, cu_chart_size, cu_num_chart,
        F
    ));
    CUDA_CHECK(cudaFree(cu_unique_chart_ids));
    CUDA_CHECK(cudaFree(cu_num_chart));
    // 3. Exclusive scan for chart face offset
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_chart_size, mesh.atlas_chart_faces_offset.ptr,
        mesh.atlas_num_charts + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        cu_chart_size, mesh.atlas_chart_faces_offset.ptr,
        mesh.atlas_num_charts + 1
    ));
    CUDA_CHECK(cudaFree(cu_chart_size));
    // 4. Expand chart ids and vertex ids
    uint64_t* cu_pack;
    CUDA_CHECK(cudaMalloc(&cu_pack, 3 * F * sizeof(uint64_t)));
    expand_chart_ids_and_vertex_ids_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_chart_ids,
        cu_sorted_face_idx,
        mesh.faces.ptr,
        F,
        cu_pack
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_sorted_chart_ids));
    CUDA_CHECK(cudaFree(cu_sorted_face_idx));
    // 5. Compress pair to construct all maps
    uint64_t* cu_inverse_pack;
    CUDA_CHECK(cudaMalloc(&cu_inverse_pack, 3 * F * sizeof(uint64_t)));
    int new_num_vertices = compress_ids(
        cu_pack,
        3 * F,
        mesh.cub_temp_storage,
        cu_inverse_pack
    );
    mesh.atlas_chart_vertex_map.resize(new_num_vertices);
    mesh.atlas_chart_vertex_offset.resize(mesh.atlas_num_charts + 1);
    unpack_vertex_ids_kernel<<<(new_num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_inverse_pack,
        new_num_vertices,
        mesh.atlas_chart_vertex_map.ptr,
        mesh.atlas_chart_vertex_offset.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_inverse_pack));
    unpack_faces_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_pack,
        F,
        mesh.atlas_chart_faces.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_pack));
}


void CuMesh::compute_charts(
    float threshold_cone_half_angle_rad, 
    int refine_iterations, 
    int global_iterations, 
    float smooth_strength,
    float area_penalty_weight,
    float perimeter_area_ratio_weight
) {
    if (this->manifold_face_adj.is_empty()) {
        this->get_manifold_face_adjacency();
    }
    if (this->face_normals.is_empty()) {
        this->compute_face_normals();
    }
    if (this->face_areas.is_empty()) {
        this->compute_face_areas();
    }

    // Initialize chart id
    size_t F = this->faces.size;
    this->atlas_chart_ids.resize(F);
    this->atlas_num_charts = F;
    arange_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        this->atlas_chart_ids.ptr,
        F
    );
    CUDA_CHECK(cudaGetLastError());

    // Main Iteration: Collapse and Refine
    int* cu_end_flag; int h_end_flag;
    CUDA_CHECK(cudaMalloc(&cu_end_flag, sizeof(int)));
    for (int i = 0; i < global_iterations; i++) {
        while (true) {
            h_end_flag = 1;
            CUDA_CHECK(cudaMemcpy(cu_end_flag, &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

            // 1. Compute chart connectivity
            get_chart_connectivity(*this);
            if (this->atlas_chart_adj.size == 0) break;

            // 2. Compute normal cones
            compute_chart_normal_cones(*this);

            // 3. Compute chart adjacency cost
            size_t E = this->atlas_chart_adj.size;
            this->edge_collapse_costs.resize(E);
            compute_chart_adjacency_cost_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                this->atlas_chart_adj.ptr,
                this->atlas_chart_normal_cones.ptr,
                this->atlas_chart_adj_length.ptr,
                this->atlas_chart_perims.ptr,
                this->atlas_chart_areas.ptr,
                area_penalty_weight,
                perimeter_area_ratio_weight,
                E,
                this->edge_collapse_costs.ptr
            );
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());

            // 4. Propagate costs
            size_t C = this->atlas_num_charts;
            this->propagated_costs.resize(C);
            propagate_cost_kernel<<<(C + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                this->atlas_chart2edge.ptr,
                this->atlas_chart2edge_offset.ptr,
                this->edge_collapse_costs.ptr,
                C,
                this->propagated_costs.ptr
            );
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());

            // 5. Collapse edges
            this->vertices_map.resize(C);      // store collapse map
            arange_kernel<<<(C + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                this->vertices_map.ptr,
                C
            );
            collapse_edges_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                this->atlas_chart_adj.ptr,
                this->edge_collapse_costs.ptr,
                this->propagated_costs.ptr,
                threshold_cone_half_angle_rad,
                E,
                this->vertices_map.ptr,
                this->atlas_chart_normal_cones.ptr,
                cu_end_flag
            );
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());

            // End of iteration
            CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_end_flag == 1) break;

            // 6. Compress chart ids
            this->atlas_num_charts = compress_ids(this->vertices_map.ptr, C, this->cub_temp_storage);
            this->temp_storage.resize(F * sizeof(int));
            index_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                this->vertices_map.ptr,
                this->atlas_chart_ids.ptr,
                F,
                reinterpret_cast<int*>(this->temp_storage.ptr)
            );
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
            swap_buffers(this->atlas_chart_ids, this->temp_storage);
        }

        // Refine charts
        for (int j = 0; j < refine_iterations; j++) {
            compute_chart_normal_cones(*this);
            this->temp_storage.resize(F * sizeof(int));
            refine_charts_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                this->atlas_chart_normal_cones.ptr,
                this->face_normals.ptr,
                this->vertices.ptr,
                this->edges.ptr,
                this->face2edge.ptr,
                this->edge2face.ptr,
                this->edge2face_offset.ptr,
                F,
                smooth_strength,
                this->atlas_chart_ids.ptr,
                reinterpret_cast<int*>(this->temp_storage.ptr)
            );
            CUDA_CHECK(cudaGetLastError());
            swap_buffers(this->atlas_chart_ids, this->temp_storage);
            this->atlas_num_charts = compress_ids(this->atlas_chart_ids.ptr, F, this->cub_temp_storage);
        }

        // After refinement, the chart may become disconnected, so we need to re-assign chart ids
        reassign_chart_ids(*this);
    }
    CUDA_CHECK(cudaFree(cu_end_flag));

    // Finalizing: calculate vmap, chart face and chart face offset
    construct_chart_mesh(*this);
}


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CuMesh::read_atlas_charts() {
    auto chart_ids = torch::empty({ static_cast<int64_t>(this->faces.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        chart_ids.data_ptr<int>(),
        this->atlas_chart_ids.ptr,
        this->faces.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    auto vertex_map = torch::empty({ static_cast<int64_t>(this->atlas_chart_vertex_map.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        vertex_map.data_ptr<int>(),
        this->atlas_chart_vertex_map.ptr,
        this->atlas_chart_vertex_map.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    auto chart_faces = torch::empty({ static_cast<int64_t>(this->atlas_chart_faces.size), 3 }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        chart_faces.data_ptr<int>(),
        this->atlas_chart_faces.ptr,
        this->atlas_chart_faces.size * 3 * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    auto chart_vertex_offset = torch::empty({ static_cast<int64_t>(this->atlas_chart_vertex_offset.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        chart_vertex_offset.data_ptr<int>(),
        this->atlas_chart_vertex_offset.ptr,
        this->atlas_chart_vertex_offset.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    auto chart_face_offset = torch::empty({ static_cast<int64_t>(this->atlas_chart_faces_offset.size) }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(
        chart_face_offset.data_ptr<int>(),
        this->atlas_chart_faces_offset.ptr,
        this->atlas_chart_faces_offset.size * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    return std::make_tuple(this->atlas_num_charts, chart_ids, vertex_map, chart_faces, chart_vertex_offset, chart_face_offset);
}


} // namespace cumesh