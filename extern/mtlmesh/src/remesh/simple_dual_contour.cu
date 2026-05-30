#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "api.h"
#include "../utils.h"
#include "../hash/hash.cuh"


template<typename T>
__device__ __forceinline__ float get_vertex_val(
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const float* __restrict__ udf,
    const size_t N_vert,
    int x, int y, int z,
    int W, int H, int D
) {
    size_t flat_idx = (size_t)x * H * D + (size_t)y * D + z;
    T key = static_cast<T>(flat_idx);
    uint32_t idx = linear_probing_lookup(hashmap_keys, hashmap_vals, key, N_vert);
    return udf[idx];
}


template<typename T>
static __global__ void simple_dual_contour_kernel(
    const size_t N_vert,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const int32_t* __restrict__ coords,
    const float* __restrict__ udf,
    float* __restrict__ out_vertices,
    int32_t* __restrict__ out_intersected
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= M) return;

    int vx = coords[thread_id * 3 + 0];
    int vy = coords[thread_id * 3 + 1];
    int vz = coords[thread_id * 3 + 2];

    float3 intersection_sum = make_float3(0.0f, 0.0f, 0.0f);
    int intersection_count = 0;

    // Traverse the 12 edges of the voxel
    // Axis X
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx, vy + u, vz + v, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + 1, vy + u, vz + v, W, H, D);

            // Calculate the intersection point
            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                // P = P1 + t * (P2 - P1)
                intersection_sum.x += (float)vx + t;
                intersection_sum.y += (float)(vy + u);
                intersection_sum.z += (float)(vz + v);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 0] = 1;
                }
                else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 0] = -1;
                }
                else {
                    out_intersected[thread_id * 3 + 0] = 0;
                }
            }
        }
    }

    // Axis Y
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy, vz + v, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy + 1, vz + v, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += (float)(vx + u);
                intersection_sum.y += (float)vy + t;
                intersection_sum.z += (float)(vz + v);
                intersection_count++;
            }
            
            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 1] = 1;
                }
                else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 1] = -1;
                }
                else {
                    out_intersected[thread_id * 3 + 1] = 0;
                }
            }
        }
    }

    // Axis Z
    #pragma unroll
    for (int u = 0; u <= 1; ++u) {
        #pragma unroll
        for (int v = 0; v <= 1; ++v) {
            float val1 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy + v, vz, W, H, D);
            float val2 = get_vertex_val(hashmap_keys, hashmap_vals, udf, N_vert, vx + u, vy + v, vz + 1, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += (float)(vx + u);
                intersection_sum.y += (float)(vy + v);
                intersection_sum.z += (float)vz + t;
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[thread_id * 3 + 2] = 1;
                }
                else if (val1 >= 0 && val2 < 0) {
                    out_intersected[thread_id * 3 + 2] = -1;
                }
                else {
                    out_intersected[thread_id * 3 + 2] = 0;
                }
            }
        }
    }

    // Calculate the mean intersection point
    if (intersection_count > 0) {
        out_vertices[thread_id * 3 + 0] = intersection_sum.x / intersection_count;
        out_vertices[thread_id * 3 + 1] = intersection_sum.y / intersection_count;
        out_vertices[thread_id * 3 + 2] = intersection_sum.z / intersection_count;
    } else {
        // Fallback: Voxel Center
        out_vertices[thread_id * 3 + 0] = (float)vx + 0.5f;
        out_vertices[thread_id * 3 + 1] = (float)vy + 0.5f;
        out_vertices[thread_id * 3 + 2] = (float)vz + 0.5f;
    }
}


/**
 * Isosurfacing a volume defined on vertices of a sparse voxel grid using a simple dual contouring algorithm.
 * Dual vertices are computed by mean of edge intersections.
 * 
 * @param hashmap_keys  [Nvert] uint32/uint64 hashmap of the vertices keys
 * @param hashmap_vals  [Nvert] uint32 tensor containing the hashmap values as vertex indices
 * @param coords        [Mvox, 3] int32 tensor containing the coordinates of the active voxels
 * @param udf           [Mvert] float tensor containing the UDF/SDF values at the vertices
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 *
 * @return              [L, 3] float tensor containing the active vertices (Dual Vertices)
                        [L, 3] int32 tensor containing the intersected edges (1: intersected, 0: not intersected)
 */
std::tuple<torch::Tensor, torch::Tensor> cumesh::simple_dual_contour(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& udf,
    int W,
    int H,
    int D
) {
    const size_t M = coords.size(0);
    const size_t N_vert = hashmap_keys.size(0);

    auto vertices = torch::empty({(long)M, 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    auto intersected = torch::empty({(long)M, 3}, torch::dtype(torch::kInt32).device(coords.device()));

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (hashmap_keys.dtype() == torch::kUInt32) {
        simple_dual_contour_kernel<<<blocks, threads>>>(
            N_vert,
            M,
            W, H, D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            udf.data_ptr<float>(),
            vertices.data_ptr<float>(),
            intersected.data_ptr<int32_t>()
        );
    } 
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        simple_dual_contour_kernel<<<blocks, threads>>>(
            N_vert,
            M,
            W, H, D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            udf.data_ptr<float>(),
            vertices.data_ptr<float>(),
            intersected.data_ptr<int32_t>()
        );
    } 
    else {
        TORCH_CHECK(false, "Unsupported hashmap data type");
    }

    CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vertices, intersected);
}
