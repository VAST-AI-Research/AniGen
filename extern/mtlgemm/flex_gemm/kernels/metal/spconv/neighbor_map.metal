#include <metal_stdlib>
using namespace metal;

#include "../hash/hash.h"

// ============================================================================
// Submanifold conv neighbor map — uint32 keys
// ============================================================================

kernel void submanifold_conv_neighbor_map_u32(
    const device uint* hashmap_keys   [[buffer(0)]],
    const device uint* hashmap_vals   [[buffer(1)]],
    const device int* coords          [[buffer(2)]],   // [M, 4]
    device uint* neighbor             [[buffer(3)]],   // [M, V]
    constant uint& hash_N             [[buffer(4)]],   // hashmap size
    constant uint& M                  [[buffer(5)]],   // num coords
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    constant int& V                   [[buffer(9)]],   // kernel volume
    constant int& Kw                  [[buffer(10)]],
    constant int& Kh                  [[buffer(11)]],
    constant int& Kd                  [[buffer(12)]],
    constant int& Dw                  [[buffer(13)]],
    constant int& Dh                  [[buffer(14)]],
    constant int& Dd                  [[buffer(15)]],
    uint tid                          [[thread_position_in_grid]]
) {
    int half_V = V / 2 + 1;
    uint idx = tid / half_V;
    if (idx >= M) return;

    int b = coords[idx * 4 + 0];
    int cx = coords[idx * 4 + 1] - Kw / 2 * Dw;
    int cy = coords[idx * 4 + 2] - Kh / 2 * Dh;
    int cz = coords[idx * 4 + 3] - Kd / 2 * Dd;
    int KhKd = Kh * Kd;
    int v = tid % half_V;

    uint value = 0xFFFFFFFFu;
    if (v == half_V - 1) {
        value = idx;
    } else {
        int kx = cx + v / KhKd * Dw;
        int ky = cy + v / Kd % Kh * Dh;
        int kz = cz + v % Kd * Dd;
        if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
            uint key = (uint)flatten_3d(b, kx, ky, kz, W, H, D);
            value = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, key, hash_N);
            if (value != 0xFFFFFFFFu) {
                neighbor[value * V + V - 1 - v] = idx;
            }
        }
    }
    neighbor[idx * V + v] = value;
}

// ============================================================================
// Submanifold conv neighbor map — uint64 keys (split hi/lo)
// ============================================================================

kernel void submanifold_conv_neighbor_map_u64(
    const device uint* keys_hi        [[buffer(0)]],
    const device uint* keys_lo        [[buffer(1)]],
    const device uint* hashmap_vals   [[buffer(2)]],
    const device int* coords          [[buffer(3)]],
    device uint* neighbor             [[buffer(4)]],
    constant uint& hash_N             [[buffer(5)]],
    constant uint& M                  [[buffer(6)]],
    constant int& W                   [[buffer(7)]],
    constant int& H                   [[buffer(8)]],
    constant int& D                   [[buffer(9)]],
    constant int& V                   [[buffer(10)]],
    constant int& Kw                  [[buffer(11)]],
    constant int& Kh                  [[buffer(12)]],
    constant int& Kd                  [[buffer(13)]],
    constant int& Dw                  [[buffer(14)]],
    constant int& Dh                  [[buffer(15)]],
    constant int& Dd                  [[buffer(16)]],
    uint tid                          [[thread_position_in_grid]]
) {
    int half_V = V / 2 + 1;
    uint idx = tid / half_V;
    if (idx >= M) return;

    int b = coords[idx * 4 + 0];
    int cx = coords[idx * 4 + 1] - Kw / 2 * Dw;
    int cy = coords[idx * 4 + 2] - Kh / 2 * Dh;
    int cz = coords[idx * 4 + 3] - Kd / 2 * Dd;
    int KhKd = Kh * Kd;
    int v = tid % half_V;

    uint value = 0xFFFFFFFFu;
    if (v == half_V - 1) {
        value = idx;
    } else {
        int kx = cx + v / KhKd * Dw;
        int ky = cy + v / Kd % Kh * Dh;
        int kz = cz + v % Kd * Dd;
        if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
            ulong key = flatten_3d(b, kx, ky, kz, W, H, D);
            value = linear_probing_lookup_u64(keys_hi, keys_lo, hashmap_vals, key, hash_N);
            if (value != 0xFFFFFFFFu) {
                neighbor[value * V + V - 1 - v] = idx;
            }
        }
    }
    neighbor[idx * V + v] = value;
}

// ============================================================================
// Submanifold conv neighbor map — uint64 keys (direct atomic_ulong path)
// Used on the MPS dispatch path to avoid the hi/lo split copy — see hash.h.
// ============================================================================

kernel void submanifold_conv_neighbor_map_u64_packed(
    const device uint* packed_keys    [[buffer(0)]],
    const device uint* hashmap_vals   [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    device uint* neighbor             [[buffer(3)]],
    constant uint& hash_N             [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    constant int& V                   [[buffer(9)]],
    constant int& Kw                  [[buffer(10)]],
    constant int& Kh                  [[buffer(11)]],
    constant int& Kd                  [[buffer(12)]],
    constant int& Dw                  [[buffer(13)]],
    constant int& Dh                  [[buffer(14)]],
    constant int& Dd                  [[buffer(15)]],
    uint tid                          [[thread_position_in_grid]]
) {
    int half_V = V / 2 + 1;
    uint idx = tid / half_V;
    if (idx >= M) return;

    int b = coords[idx * 4 + 0];
    int cx = coords[idx * 4 + 1] - Kw / 2 * Dw;
    int cy = coords[idx * 4 + 2] - Kh / 2 * Dh;
    int cz = coords[idx * 4 + 3] - Kd / 2 * Dd;
    int KhKd = Kh * Kd;
    int v = tid % half_V;

    uint value = 0xFFFFFFFFu;
    if (v == half_V - 1) {
        value = idx;
    } else {
        int kx = cx + v / KhKd * Dw;
        int ky = cy + v / Kd % Kh * Dh;
        int kz = cz + v % Kd * Dd;
        if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
            ulong key = flatten_3d(b, kx, ky, kz, W, H, D);
            value = linear_probing_lookup_u64_packed(packed_keys, hashmap_vals, key, hash_N);
            if (value != 0xFFFFFFFFu) {
                neighbor[value * V + V - 1 - v] = idx;
            }
        }
    }
    neighbor[idx * V + v] = value;
}

// ============================================================================
// Neighbor map → gray/binary code + transpose
// Uses threadgroup shared memory for the transpose
// ============================================================================

kernel void neighbor_map_gray_code(
    const device uint* neighbor_map   [[buffer(0)]],   // [N, V]
    device int* gray_code             [[buffer(1)]],   // [N]
    device int* binary_code           [[buffer(2)]],   // [N]
    device uint* neigh_map_T          [[buffer(3)]],   // [V * N]
    device int* neigh_mask_T          [[buffer(4)]],   // [V * N]
    constant uint& N                  [[buffer(5)]],
    constant uint& V                  [[buffer(6)]],
    threadgroup uint* shared          [[threadgroup(0)]],  // BLOCK_SIZE * V
    uint tid                          [[thread_position_in_grid]],
    uint lid                          [[thread_index_in_threadgroup]],
    uint gid                          [[threadgroup_position_in_grid]]
) {
    uint n_base = gid * 256;  // BLOCK_SIZE = 256
    uint len_n = min(256u, N - n_base);
    uint total_len = len_n * V;

    // Load neighbor map into shared memory
    uint idx = lid;
    while (idx < total_len) {
        shared[idx] = neighbor_map[n_base * V + idx];
        idx += 256;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transpose neighbor map
    idx = lid;
    while (idx < total_len) {
        uint v = idx / len_n;
        uint n = idx % len_n;
        uint tmp = shared[n * V + v];
        neigh_map_T[v * N + n + n_base] = tmp;
        neigh_mask_T[v * N + n + n_base] = (tmp != 0xFFFFFFFFu) ? 1 : 0;
        idx += 256;
    }

    if (tid < N) {
        // Build gray code
        uint gray = 0;
        for (uint v = 0; v < V; v++) {
            uint nb = shared[lid * V + v];
            if (nb != 0xFFFFFFFFu) gray += 1u << v;
        }
        // Gray → binary code
        uint binary = gray;
        for (uint v = 1; v < V; v++) {
            binary ^= gray >> v;
        }
        gray_code[tid] = (int)gray;
        binary_code[tid] = (int)binary;
    }
}

// ============================================================================
// Gather idx/val/seg from prefix sum
// ============================================================================

kernel void gather_idx_val_seg_from_prefix_sum(
    const device int* prefix_sum      [[buffer(0)]],   // [V * N] cumsum of mask
    const device uint* values         [[buffer(1)]],   // [V * N] transposed neighbor map
    device uint* idx_out              [[buffer(2)]],   // valid signal o
    device uint* val_out              [[buffer(3)]],   // valid signal i
    device uint* seg_out              [[buffer(4)]],   // [V + 1]
    constant uint& N                  [[buffer(5)]],
    constant uint& V                  [[buffer(6)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid < N * V) {
        uint value = values[tid];
        if (value != 0xFFFFFFFFu) {
            int to = prefix_sum[tid] - 1;
            idx_out[to] = tid % N;
            val_out[to] = value;
        }
    }
    if (tid / (V + 1) == 0) {
        seg_out[tid] = (tid == 0) ? 0 : (uint)prefix_sum[tid * N - 1];
    }
}

// ============================================================================
// Reduce gray codes in blocks (OR reduction)
// ============================================================================

kernel void reduce_code(
    const device int* gray_code       [[buffer(0)]],   // [N]
    const device long* sorted_idx     [[buffer(1)]],   // [N]
    device int* reduced_code          [[buffer(2)]],   // [num_blocks]
    device int* seglen                [[buffer(3)]],   // [num_blocks + 1]
    constant uint& N                  [[buffer(4)]],
    constant int& block_dim           [[buffer(5)]],
    threadgroup int* buf              [[threadgroup(0)]],  // [256]
    uint tid                          [[thread_position_in_grid]],
    uint lid                          [[thread_index_in_threadgroup]],
    uint gid                          [[threadgroup_position_in_grid]]
) {
    int seg_per_block = 256 * 2 / block_dim;
    int seg_id = lid * 2 / block_dim;


    // Load gray code, two elements per thread
    uint n = 2 * tid;
    int e0 = 0, e1 = 0;
    if (n < N) e0 = gray_code[sorted_idx[n]];
    if (n + 1 < N) e1 = gray_code[sorted_idx[n + 1]];
    buf[seg_id + (lid % (block_dim / 2)) * seg_per_block] = e0 | e1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce buffer — iterative halving
    // Count iterations needed (log2-based)
    int total_iters = 0;
    {
        int tmp = block_dim;
        while (tmp > 1) { total_iters++; tmp >>= 1; }
        total_iters -= 1; // __ffs(block_dim) - 2 equivalent
    }

    for (int i = 0; i < total_iters; i++) {
        int cur_len = 256 >> (i + 1);
        if ((int)lid < cur_len) {
            buf[lid] |= buf[lid + cur_len];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store reduced code and segment length
    if ((int)lid < seg_per_block && (int)(gid * seg_per_block + lid) < (int)((N + block_dim - 1) / block_dim)) {
        reduced_code[gid * seg_per_block + lid] = buf[lid];
        seglen[gid * seg_per_block + lid + 1] = popcount((uint)buf[lid]);
        if (tid == 0) seglen[0] = 0;
    }
}

// ============================================================================
// Scatter reduced code → valid kernel indices
// ============================================================================

kernel void scatter_reduced_code(
    const device int* reduced_code    [[buffer(0)]],
    const device int* seglen          [[buffer(1)]],   // cumsum'd
    device int* idx_out               [[buffer(2)]],
    constant uint& num_blocks         [[buffer(3)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= num_blocks) return;
    int code = reduced_code[tid];
    int seg_start = seglen[tid];
    int seg_end = seglen[tid + 1];
    int seg_len = seg_end - seg_start;
    for (int i = 0; i < seg_len; i++) {
        int pos = ctz(code);  // __ffs(x) - 1 equivalent
        idx_out[seg_start + i] = pos;
        code &= ~(1 << pos);
    }
}
