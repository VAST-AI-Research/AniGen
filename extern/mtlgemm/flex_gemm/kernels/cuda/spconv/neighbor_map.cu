#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "neighbor_map.h"
#include "../hash/api.h"
#include "../hash/hash.cuh"


namespace flex_gemm {
namespace spconv {

/**
 * Lookup sparse submanifold convolution neighbor map with hashmap
 * 
 * @param N             number of elements in the hashmap
 * @param M             number of 3d coordinates
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param V             the volume of the kernel
 * @param Kw            the number of width kernel dimensions
 * @param Kh            the number of height kernel dimensions
 * @param Kd            the number of depth kernel dimensions
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [M, 4] int32 tensor containing the keys to be looked up
 * @param neighbor      [M, Kw * Kh * Kd] uint32 tensor containing the submanifold convolution nerbor map
 */
template<typename T>
__global__ void hashmap_lookup_submanifold_conv_neighbour_map_cuda_kernel(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    const int V,
    const int Kw,
    const int Kh,
    const int Kd,
    const int Dw,
    const int Dh,
    const int Dd,
    const T* __restrict__  hashmap_keys,
    const uint32_t* __restrict__  hashmap_vals,
    const int32_t* __restrict__  coords,
    uint32_t* __restrict__ neighbor
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int half_V = V / 2 + 1;
    uint32_t idx = static_cast<uint32_t>(thread_id / half_V);
    if (idx < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[idx];
        int b = coord.x;
        int x = coord.y - Kw / 2 * Dw;
        int y = coord.z - Kh / 2 * Dh;
        int z = coord.w - Kd / 2 * Dd;
        int KhKd = Kh * Kd;
        int v = thread_id % half_V;
        
        uint32_t value = std::numeric_limits<uint32_t>::max();
        if (v == half_V - 1) {
            value = idx;
        }
        else {
            int kx = x + v / KhKd * Dw;
            int ky = y + v / Kd % Kh * Dh;
            int kz = z + v % Kd * Dd;
            if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
                size_t flat_idx = (size_t)b * W * H * D + (size_t)kx * H * D + (size_t)ky * D + (size_t)kz;
                T key = static_cast<T>(flat_idx);
                value = flex_gemm::hash::linear_probing_lookup(hashmap_keys, hashmap_vals, key, N);
                if (value != std::numeric_limits<uint32_t>::max()) {
                    neighbor[value * V + V - 1 - v] = idx;
                }
            }
        }
        neighbor[idx * V + v] = value;
    }
}


/**
 * Build sparse submanifold convolution neighbor map with hashmap
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [M, 4] int32 tensor containing the keys to be looked up
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param Kw            the number of width kernel dimensions
 * @param Kh            the number of height kernel dimensions
 * @param Kd            the number of depth kernel dimensions
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 *  
 * @return              [M, Kw * Kh * Kd] uint32 tensor containing the submanifold convolution neighbor map
 */
torch::Tensor hashmap_build_submanifold_conv_neighbour_map_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    int W,
    int H,
    int D,
    int Kw,
    int Kh,
    int Kd,
    int Dw,
    int Dh,
    int Dd
) {
    // Allocate output tensor
    int V = Kw * Kh * Kd;

    // Insert 3D coordinates into the hashmap
    flex_gemm::hash::hashmap_insert_3d_idx_as_val_cuda(
        hashmap_keys,
        hashmap_vals,
        coords,
        W, H, D
    );

    auto neighbor = torch::full({coords.size(0), V}, std::numeric_limits<uint32_t>::max(), torch::dtype(torch::kUInt32).device(hashmap_keys.device()));

    if (hashmap_keys.dtype() == torch::kUInt32) {
        hashmap_lookup_submanifold_conv_neighbour_map_cuda_kernel<<<
            (coords.size(0) * (V / 2 + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            coords.size(0),
            W, H, D, V,
            Kw, Kh, Kd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>()
        );
    }
    else if (hashmap_keys.dtype() == torch::kUInt64) {
        hashmap_lookup_submanifold_conv_neighbour_map_cuda_kernel<<<
            (coords.size(0) * (V / 2 + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            coords.size(0),
            W, H, D, V,
            Kw, Kh, Kd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_vals.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>()
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported hashmap dtype. Expect uint32 or uint64.");
    }

    return neighbor;
}


__global__ void neighbor_map_to_gray_binary_code_and_T_map_cuda_kernel(
    const uint32_t N,
    const uint32_t V,
    const uint32_t* __restrict__ neighbor_map,
    int32_t* __restrict__ gray_code,
    int32_t* __restrict__ binary_code,
    uint32_t* __restrict__ neigh_map_T,
    int32_t* __restrict__ neigh_mask_T
) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_base = blockIdx.x * blockDim.x;
    extern __shared__ uint32_t neigh_map[];

    // Load neighbor map into shared memory
    int len_n = min(BLOCK_SIZE, N - n_base);
    int total_len = len_n * V;
    if (total_len % 4 == 0) {   // 128-bit load
        int idx = threadIdx.x * 4;
        while (idx < total_len) {
            *(int4*)&neigh_map[idx] = *(int4*)&neighbor_map[n_base * V + idx];
            idx += BLOCK_SIZE * 4;
        }
    }
    else if (total_len % 2 == 0) {   // 64-bit load
        int idx = threadIdx.x * 2;
        while (idx < total_len) {
            *(int2*)&neigh_map[idx] = *(int2*)&neighbor_map[n_base * V + idx];
            idx += BLOCK_SIZE * 2;
        }
    }
    else {   // 32-bit load
        int idx = threadIdx.x;
        while (idx < total_len) {
            neigh_map[idx] = neighbor_map[n_base * V + idx];
            idx += BLOCK_SIZE;
        }
    }

    __syncthreads();

    // Transpose neighbor map
    // 32-bit transpose
    int idx = threadIdx.x;
    while (idx < total_len) {
        int v = idx / len_n;
        int n = idx % len_n;
        uint32_t tmp = neigh_map[n * V + v];
        neigh_map_T[v * N + n + n_base] = tmp;
        tmp = tmp != std::numeric_limits<uint32_t>::max();
        neigh_mask_T[v * N + n + n_base] = tmp;
        idx += BLOCK_SIZE;
    }

    if (thread_id < N) {
        // Build gray code
        uint32_t gray = 0;
        for (uint32_t v = 0; v < V; v++) {
            uint32_t neighbor = neigh_map[threadIdx.x * V + v];
            if (neighbor != std::numeric_limits<uint32_t>::max()) gray += 1 << v;
        }
        // Gray code to binary code
        uint32_t binary = gray;
        for (uint32_t v = 1; v < V; v++) {
            binary ^= gray >> v;
        }
        // Store gray and binary code
        gray_code[thread_id] = gray;
        binary_code[thread_id] = binary;
    }
}


__global__ void gather_idx_val_seg_from_prefix_sum_cuda_kernel(
    const uint32_t N,
    const uint32_t V,
    const int32_t* __restrict__ prefix_sum,
    const uint32_t* __restrict__ values,
    uint32_t* __restrict__ idx,
    uint32_t* __restrict__ val,
    uint32_t* __restrict__ seg
) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N * V) {
        int value = values[thread_id];
        if (value != 0xffffffff) {
            int to = prefix_sum[thread_id] - 1;
            idx[to] = thread_id % N;
            val[to] = value;
        }
    }
    if (thread_id / (V + 1) == 0) {
        seg[thread_id] = thread_id == 0 ? 0 : prefix_sum[thread_id * N - 1];
    }
}


/**
 * Convert neighbor map to gray and binary code
 * 
 * @param neighbor_map     [N, V] uint32 tensor containing the neighbor map
 * 
 * @return                 [N] gray code
 *                         [N] sorted idx
 *                         [N] valid signal idx for input tensor
 *                         [N] valid signal idx for output tensor
 *                         [V+1] valid signal segment
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_1(
    const torch::Tensor& neighbor_map
) {
    const int64_t N = neighbor_map.size(0);
    const int64_t V = neighbor_map.size(1);

    // Allocate output tensors
    auto gray_code = torch::empty({N}, torch::dtype(torch::kInt32).device(neighbor_map.device()));
    auto binary_code = torch::empty({N}, torch::dtype(torch::kInt32).device(neighbor_map.device()));
    auto neigh_mask_T = torch::empty({V * N}, torch::dtype(torch::kInt32).device(neighbor_map.device()));
    auto neigh_map_T = torch::empty({V * N}, torch::dtype(torch::kUInt32).device(neighbor_map.device()));

    // Convert neighbor map to gray and binary code
    neighbor_map_to_gray_binary_code_and_T_map_cuda_kernel<<<
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        BLOCK_SIZE * V * sizeof(uint32_t)
    >>>(
        N,
        V,
        neighbor_map.data_ptr<uint32_t>(),
        gray_code.data_ptr<int32_t>(),
        binary_code.data_ptr<int32_t>(),
        neigh_map_T.data_ptr<uint32_t>(),
        neigh_mask_T.data_ptr<int32_t>()
    );

    auto sorted_idx = torch::argsort(binary_code);

    // Indexing from mask
    auto prefix_sum_neighbor_mask = torch::cumsum(neigh_mask_T, 0, torch::kInt32);
    auto total_valid_signal = prefix_sum_neighbor_mask[-1].item<int32_t>();
    auto valid_signal_i = torch::empty({total_valid_signal}, torch::dtype(torch::kUInt32).device(neighbor_map.device()));
    auto valid_signal_o = torch::empty({total_valid_signal}, torch::dtype(torch::kUInt32).device(neighbor_map.device()));
    auto valid_signal_seg = torch::empty({V + 1}, torch::dtype(torch::kUInt32).device(neighbor_map.device()));
    gather_idx_val_seg_from_prefix_sum_cuda_kernel<<<
        (N * V + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        N,
        V,
        prefix_sum_neighbor_mask.data_ptr<int32_t>(),
        neigh_map_T.data_ptr<uint32_t>(),
        valid_signal_o.data_ptr<uint32_t>(),
        valid_signal_i.data_ptr<uint32_t>(),
        valid_signal_seg.data_ptr<uint32_t>()
    );

    return std::make_tuple(gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg);
}


__global__ void reduce_code_cuda_kernel(
    const uint32_t N,
    const int block_dim,
    const int32_t* __restrict__ gray_code,
    const int64_t* __restrict__ sorted_idx,
    int32_t* __restrict__ reduced_code,
    int32_t* __restrict__ seglen
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int seg_per_block = BLOCK_SIZE * 2 / block_dim;
    int seg_id = threadIdx.x * 2 / block_dim;
    int inner_id = threadIdx.x % (block_dim / 2);
    __shared__ int32_t buf[BLOCK_SIZE];

    // Load gray code into buffer, two elements per thread
    int n = 2 * thread_id, e0 = 0, e1 = 0;
    if (n < N) e0 = gray_code[sorted_idx[n]];
    if (n + 1 < N) e1 = gray_code[sorted_idx[n + 1]];
    buf[seg_id + inner_id * seg_per_block] = e0 | e1;
    __syncthreads();

    // Reduce buffer
    int total_iters = __ffs(block_dim) - 2;
    int iters_blockwise = min(__ffs(BLOCK_SIZE / warpSize) - 2, total_iters);
    int iters_warpwise = total_iters - iters_blockwise;

    #pragma unroll
    for (int i = 0; i < iters_blockwise; i++) {
        int cur_len = BLOCK_SIZE >> (i + 1);
        if (threadIdx.x < cur_len) {
            buf[threadIdx.x] |= buf[threadIdx.x + cur_len];
        }
        __syncthreads();
    }
    
    if (iters_warpwise > 0 && threadIdx.x < warpSize) {
        #pragma unroll
        for (int i = 0; i < iters_warpwise; i++) {
            int cur_len = warpSize >> i;
            buf[threadIdx.x] |= buf[threadIdx.x + cur_len];
            __syncwarp();
        }
    }

    // Store reduced code and segment length
    if (threadIdx.x < seg_per_block && blockIdx.x * seg_per_block + threadIdx.x < (N + block_dim - 1) / block_dim) {
        reduced_code[blockIdx.x * seg_per_block + threadIdx.x] = buf[threadIdx.x];
        seglen[blockIdx.x * seg_per_block + threadIdx.x + 1] = __popc(buf[threadIdx.x]);
        if (thread_id == 0) seglen[0] = 0;
    }
}
    
    
__global__ void scatter_reduced_code_cuda_kernel(
    const int num_blocks,
    const int32_t* __restrict__ reduced_code,
    const int32_t* __restrict__ seglen,
    int32_t* __restrict__ idx 
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < num_blocks) {
        int code = reduced_code[thread_id];
        int seg_start = seglen[thread_id];
        int seg_end = seglen[thread_id + 1];
        int seg_len = seg_end - seg_start;
        for (int i = 0; i < seg_len; i++) {
            int pos = __ffs(code) - 1;
            idx[seg_start + i] = pos;
            code &= ~(1 << pos);
        }
    }
}


/**
 * Get valid kernel indices for masked implicit gemm
 * 
 * @param gray_code     [N] gray code
 * @param sorted_idx    [N] sorted idx
 * @param block_size    the block size of CUDA kernel (must be power of 2)
 * 
 * @return              [L] uint32 tensor containing the valid kernel indices
 *                      [num_blocks + 1] uint32 tensor containing the segment of valid kernel indices
 */
std::tuple<torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_2(
    const torch::Tensor& gray_code,
    const torch::Tensor& sorted_idx,
    int block_size
) {
    const uint32_t N = gray_code.size(0);
    
    // Reduce gray code to reduced code and segment length
    auto num_blocks = (N + block_size - 1) / block_size;
    auto reduced_code = torch::empty({num_blocks}, torch::dtype(torch::kInt32).device(gray_code.device()));
    auto seglen = torch::empty({num_blocks + 1}, torch::dtype(torch::kInt32).device(gray_code.device()));
    reduce_code_cuda_kernel<<<
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        N,
        block_size,
        gray_code.data_ptr<int32_t>(),
        sorted_idx.data_ptr<int64_t>(),
        reduced_code.data_ptr<int32_t>(),
        seglen.data_ptr<int32_t>()
    );

    seglen = torch::cumsum(seglen, 0, torch::kInt32);

    // Scatter reduced code to valid kernel indices
    auto valid_kernel_idx = torch::empty({seglen[-1].item<int32_t>()}, torch::dtype(torch::kInt32).device(gray_code.device()));
    scatter_reduced_code_cuda_kernel<<<
        (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        num_blocks,
        reduced_code.data_ptr<int32_t>(),
        seglen.data_ptr<int32_t>(),
        valid_kernel_idx.data_ptr<int32_t>()
    );

    return std::make_tuple(valid_kernel_idx, seglen);
}

} // namespace spconv
} // namespace flex_gemm
