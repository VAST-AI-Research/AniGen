#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "api.h"
#include "../utils.h"
#include "../hash/api.h"
#include "../hash/hash.cuh"


template<typename T>
static __global__ void get_vertex_num(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const int32_t* __restrict__ coords,
    int* __restrict__ num_vertices
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= M) return;

    int num = 1;        // include the current voxel
    
    int x = coords[3 * thread_id + 0];
    int y = coords[3 * thread_id + 1];
    int z = coords[3 * thread_id + 2];

    size_t flat_idx;
    T key;

    #pragma unroll
    for (int i = 0; i <= 1; i++) {
        #pragma unroll
        for (int j = 0; j <= 1; j++) {
            #pragma unroll
            for (int k = 0; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                int xx = x + i;
                int yy = y + j;
                int zz = z + k;
                if (xx >= W || yy >= H || zz >= D) {
                    num++;
                    continue;
                }
                flat_idx = (size_t)(xx * H + yy) * D + zz;
                key = static_cast<T>(flat_idx);
                if (linear_probing_lookup(hashmap_keys, hashmap_vals, key, N) == std::numeric_limits<uint32_t>::max()) {
                    num++;
                }
            }
        }
    }

    num_vertices[thread_id] = num;
}


template<typename T>
static __global__ void set_vertex(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const int32_t* __restrict__ coords,
    const int* __restrict__ vertices_offset,
    int* __restrict__ vertices
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= M) return;
    
    int x = coords[3 * thread_id + 0];
    int y = coords[3 * thread_id + 1];
    int z = coords[3 * thread_id + 2];
    int ptr_start = vertices_offset[thread_id];
    vertices[3 * ptr_start + 0] = x;
    vertices[3 * ptr_start + 1] = y;
    vertices[3 * ptr_start + 2] = z;
    ptr_start++;

    size_t flat_idx;
    T key;

    #pragma unroll
    for (int i = 0; i <= 1; i++) {
        #pragma unroll
        for (int j = 0; j <= 1; j++) {
            #pragma unroll
            for (int k = 0; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                int xx = x + i;
                int yy = y + j;
                int zz = z + k;
                if (xx >= W || yy >= H || zz >= D) {
                    vertices[3 * ptr_start + 0] = xx;
                    vertices[3 * ptr_start + 1] = yy;
                    vertices[3 * ptr_start + 2] = zz;
                    ptr_start++;
                    continue;
                }
                flat_idx = (size_t)(xx * H + yy) * D + zz;
                key = static_cast<T>(flat_idx);
                if (linear_probing_lookup(hashmap_keys, hashmap_vals, key, N) == std::numeric_limits<uint32_t>::max()) {
                    vertices[3 * ptr_start + 0] = xx;
                    vertices[3 * ptr_start + 1] = yy;
                    vertices[3 * ptr_start + 2] = zz;
                    ptr_start++;
                }
            }
        }
    }
}


/**
 * Get the active vetices of a sparse voxel grid
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as voxel indices
 * @param coords        [M, 3] int32 tensor containing the coordinates of the active voxels
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 *  
 * @return              [L, 3] int32 tensor containing the active vertices
 */
torch::Tensor cumesh::get_sparse_voxel_grid_active_vertices(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const int W,
    const int H,
    const int D
) {
    // Handle empty input - return early to avoid launching kernels with 0 blocks
    size_t M = coords.size(0);
    if (M == 0) {
        return torch::empty({0, 3}, torch::dtype(torch::kInt32).device(hashmap_keys.device()));
    }

    // Get the number of active vertices for each voxel
    size_t N = hashmap_keys.size(0);
    int* num_vertices;
    CUDA_CHECK(cudaMalloc(&num_vertices, (M + 1) * sizeof(int)));
    if (hashmap_keys.dtype() == torch::kUInt32) {
        get_vertex_num<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            N,
            M,
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            num_vertices
        );
    } else if (hashmap_keys.dtype() == torch::kUInt64) {
        get_vertex_num<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            N,
            M,
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            num_vertices
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
    CUDA_CHECK(cudaGetLastError());

    // Compute the offset
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, num_vertices, M + 1);
    void* d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, num_vertices, M + 1);
    CUDA_CHECK(cudaFree(d_temp_storage));
    int total_vertices;
    CUDA_CHECK(cudaMemcpy(&total_vertices, num_vertices + M, sizeof(int), cudaMemcpyDeviceToHost));

    // Set the active vertices for each voxel
    auto vertices = torch::empty({total_vertices, 3}, torch::dtype(torch::kInt32).device(hashmap_keys.device()));
    if (hashmap_keys.dtype() == torch::kUInt32) {
        set_vertex<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            N,
            M,
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            num_vertices,
            vertices.data_ptr<int32_t>()
        );
    }
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        set_vertex<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            N,
            M,
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            num_vertices,
            vertices.data_ptr<int32_t>()
        );
    }
    CUDA_CHECK(cudaGetLastError());

    // Free the temporary memory
    CUDA_CHECK(cudaFree(num_vertices));

    return vertices;
}
