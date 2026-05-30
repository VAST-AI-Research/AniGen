#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "grid_sample.h"
#include "../hash/api.h"
#include "../hash/hash.cuh"


namespace flex_gemm {
namespace grid_sample {

/**
 * Lookup grid sample neighbor map using hashmap
 * 
 * @param N             number of elements in the hashmap
 * @param M             number of query points
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param query         [B, L, 3] float tensor containing the query points
 * @param neighbor      [B, L] uint32 tensor containing the neighboring indices
 */
template<typename T>
__global__ void hashmap_lookup_grid_sample_3d_nearest_neighbor_map_kernel(
    const uint32_t N,
    const uint32_t B,
    const uint32_t L,
    const int W,
    const int H,
    const int D,
    const T* __restrict__  hashmap_keys,
    const uint32_t* __restrict__  hashmap_vals,
    const float* __restrict__  query,
    uint32_t* __restrict__ neighbor
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < B * L) {
        int b = thread_id / L;
        int x = static_cast<int>(query[3 * thread_id]);
        int y = static_cast<int>(query[3 * thread_id + 1]);
        int z = static_cast<int>(query[3 * thread_id + 2]);

        if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
            size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + (size_t)z;
            T key = static_cast<T>(flat_idx);
            uint32_t value = flex_gemm::hash::linear_probing_lookup(hashmap_keys, hashmap_vals, key, N);
            if (value != std::numeric_limits<uint32_t>::max()) {
                neighbor[thread_id] = value;
            }
        }
    }
}


/**
 * Build the neighbor map for grid sample with nearest interpolation
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [N, 4] int32 tensor containing the coordinates of input features
 * @param grid          [B, L, 3] float tensor containing the grid to be sampled
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * 
 * @return              [B, L] uint32 tensor containing the neighboring indices
 */
torch::Tensor hashmap_build_grid_sample_3d_nearest_neighbor_map(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    const int W,
    const int H,
    const int D
) {
    // Allocate output tensor
    auto neighbor = torch::full({grid.size(0), grid.size(1)}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(hashmap_keys.device()));

    // Insert 3D coordinates into the hashmap
    flex_gemm::hash::hashmap_insert_3d_idx_as_val_cuda(
        hashmap_keys,
        hashmap_vals,
        coords,
        W, H, D
    );

    // Lookup sparse submanifold convolution neighbor map with hashmap
    if (hashmap_keys.dtype() == torch::kUInt32) {
        hashmap_lookup_grid_sample_3d_nearest_neighbor_map_kernel<<<
            (grid.size(0) * grid.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            grid.size(0),
            grid.size(1),
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            grid.data_ptr<float>(),
            neighbor.data_ptr<uint32_t>()
        );
    }
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        hashmap_lookup_grid_sample_3d_nearest_neighbor_map_kernel<<<
            (grid.size(0) * grid.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            grid.size(0),
            grid.size(1),
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            grid.data_ptr<float>(),
            neighbor.data_ptr<uint32_t>()
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported hashmap key type");
    }

    return neighbor;
}


/**
 * Lookup grid sample neighbor map and weights using hashmap
 * 
 * @param N             number of elements in the hashmap
 * @param M             number of query points
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param query         [B, L, 3] float tensor containing the query points
 * @param neighbor      [B, L, 8] uint32 tensor containing the neighboring indices
 * @param weight        [B, L, 8] float tensor containing the weights for each neighbor
 */
template<typename T>
__global__ void hashmap_lookup_grid_sample_3d_trilinear_neighbor_map_weight_kernel(
    const uint32_t N,
    const uint32_t B,
    const uint32_t L,
    const int W,
    const int H,
    const int D,
    const T* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_vals,
    const float* __restrict__ query,
    uint32_t* __restrict__ neighbor,
    float* __restrict__ weight
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < B * L) {
        int b = thread_id / L;
        float3 q = { query[3 * thread_id], query[3 * thread_id + 1], query[3 * thread_id + 2] };
        uint32_t n[8] = { std::numeric_limits<uint32_t>::max() };
        float w[8] = { 0.0f };
        float w_sum = 0.0f;

        int base_x = static_cast<int>(floor(q.x - 0.5f));
        int base_y = static_cast<int>(floor(q.y - 0.5f));
        int base_z = static_cast<int>(floor(q.z - 0.5f));

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int x = base_x + (i & 1);
            int y = base_y + ((i >> 1) & 1);
            int z = base_z + ((i >> 2) & 1);
            if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
                size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + (size_t)z;
                T key = static_cast<T>(flat_idx);
                uint32_t value = flex_gemm::hash::linear_probing_lookup(hashmap_keys, hashmap_vals, key, N);
                if (value != std::numeric_limits<uint32_t>::max()) {
                    n[i] = value;
                    w[i] = (1 - abs(q.x - x - 0.5f)) * (1 - abs(q.y - y - 0.5f)) * (1 - abs(q.z - z - 0.5f));
                    w_sum += w[i];
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            w[i] /= max(w_sum, 1e-12f);
        }
        
        reinterpret_cast<uint4*>(neighbor + 8 * thread_id)[0] = *reinterpret_cast<const uint4*>(&n[0]);
        reinterpret_cast<uint4*>(neighbor + 8 * thread_id)[1] = *reinterpret_cast<const uint4*>(&n[4]);
        reinterpret_cast<float4*>(weight + 8 * thread_id)[0] = *reinterpret_cast<const float4*>(&w[0]);
        reinterpret_cast<float4*>(weight + 8 * thread_id)[1] = *reinterpret_cast<const float4*>(&w[4]);
    }
}


/**
 * Build the neighbor map and weights for grid sample with trilinear interpolation
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [N, 4] int32 tensor containing the coordinates of input features
 * @param grid          [B, L, 3] float tensor containing the grid to be sampled
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * 
 * @return
 *      [B, L, 8] uint32 tensor containing the neighboring indices
 *      [B, L, 8] float tensor containing the weights
 */
std::tuple<torch::Tensor, torch::Tensor> hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    const int W,
    const int H,
    const int D
) {
    // Allocate output tensor
    auto neighbor = torch::full({grid.size(0), grid.size(1), 8}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(hashmap_keys.device()));
    auto weight = torch::zeros({grid.size(0), grid.size(1), 8}, torch::dtype(torch::kFloat32).device(hashmap_keys.device()));

    // Insert 3D coordinates into the hashmap
    flex_gemm::hash::hashmap_insert_3d_idx_as_val_cuda(
        hashmap_keys,
        hashmap_vals,
        coords,
        W, H, D
    );

    // Lookup sparse submanifold convolution neighbor map with hashmap
    if (hashmap_keys.dtype() == torch::kUInt32) {
        hashmap_lookup_grid_sample_3d_trilinear_neighbor_map_weight_kernel<<<
            (grid.size(0) * grid.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            grid.size(0),
            grid.size(1),
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            grid.data_ptr<float>(),
            neighbor.data_ptr<uint32_t>(),
            weight.data_ptr<float>()
        );
    }
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        hashmap_lookup_grid_sample_3d_trilinear_neighbor_map_weight_kernel<<<
            (grid.size(0) * grid.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            grid.size(0),
            grid.size(1),
            W,
            H,
            D,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            grid.data_ptr<float>(),
            neighbor.data_ptr<uint32_t>(),
            weight.data_ptr<float>()
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported hashmap key type");
    }

    return std::make_tuple(neighbor, weight);
}

} // namespace grid_sample
} // namespace flex_gemm
