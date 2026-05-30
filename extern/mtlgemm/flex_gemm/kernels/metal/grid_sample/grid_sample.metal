#include <metal_stdlib>
using namespace metal;

#include "../hash/hash.h"

// ============================================================================
// Grid sample nearest neighbor — uint32 keys
// ============================================================================

kernel void grid_sample_nearest_u32(
    const device uint* hashmap_keys   [[buffer(0)]],
    const device uint* hashmap_vals   [[buffer(1)]],
    const device float* query         [[buffer(2)]],  // [B*L, 3]
    device uint* neighbor             [[buffer(3)]],  // [B*L]
    constant uint& N                  [[buffer(4)]],  // hashmap size
    constant uint& B                  [[buffer(5)]],
    constant uint& L                  [[buffer(6)]],
    constant int& W                   [[buffer(7)]],
    constant int& H                   [[buffer(8)]],
    constant int& D                   [[buffer(9)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= B * L) return;
    int b = tid / L;
    int x = (int)query[3 * tid];
    int y = (int)query[3 * tid + 1];
    int z = (int)query[3 * tid + 2];

    if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
        uint key = (uint)flatten_3d(b, x, y, z, W, H, D);
        uint value = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, key, N);
        if (value != 0xFFFFFFFFu) {
            neighbor[tid] = value;
        }
    }
}

// ============================================================================
// Grid sample nearest neighbor — uint64 keys (split hi/lo)
// ============================================================================

kernel void grid_sample_nearest_u64(
    const device uint* keys_hi        [[buffer(0)]],
    const device uint* keys_lo        [[buffer(1)]],
    const device uint* hashmap_vals   [[buffer(2)]],
    const device float* query         [[buffer(3)]],
    device uint* neighbor             [[buffer(4)]],
    constant uint& N                  [[buffer(5)]],
    constant uint& B                  [[buffer(6)]],
    constant uint& L                  [[buffer(7)]],
    constant int& W                   [[buffer(8)]],
    constant int& H                   [[buffer(9)]],
    constant int& D                   [[buffer(10)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= B * L) return;
    int b = tid / L;
    int x = (int)query[3 * tid];
    int y = (int)query[3 * tid + 1];
    int z = (int)query[3 * tid + 2];

    if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
        ulong key = flatten_3d(b, x, y, z, W, H, D);
        uint value = linear_probing_lookup_u64(keys_hi, keys_lo, hashmap_vals, key, N);
        if (value != 0xFFFFFFFFu) {
            neighbor[tid] = value;
        }
    }
}

// ============================================================================
// Grid sample trilinear — uint32 keys
// ============================================================================

kernel void grid_sample_trilinear_u32(
    const device uint* hashmap_keys   [[buffer(0)]],
    const device uint* hashmap_vals   [[buffer(1)]],
    const device float* query         [[buffer(2)]],
    device uint* neighbor             [[buffer(3)]],  // [B*L, 8]
    device float* weight              [[buffer(4)]],  // [B*L, 8]
    constant uint& N                  [[buffer(5)]],
    constant uint& B                  [[buffer(6)]],
    constant uint& L                  [[buffer(7)]],
    constant int& W                   [[buffer(8)]],
    constant int& H                   [[buffer(9)]],
    constant int& D                   [[buffer(10)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= B * L) return;
    int b = tid / L;
    float qx = query[3 * tid];
    float qy = query[3 * tid + 1];
    float qz = query[3 * tid + 2];

    uint n[8];
    float w[8];
    float w_sum = 0.0f;

    for (int i = 0; i < 8; i++) {
        n[i] = 0xFFFFFFFFu;
        w[i] = 0.0f;
    }

    int base_x = (int)floor(qx - 0.5f);
    int base_y = (int)floor(qy - 0.5f);
    int base_z = (int)floor(qz - 0.5f);

    for (int i = 0; i < 8; i++) {
        int x = base_x + (i & 1);
        int y = base_y + ((i >> 1) & 1);
        int z = base_z + ((i >> 2) & 1);
        if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
            uint key = (uint)flatten_3d(b, x, y, z, W, H, D);
            uint value = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, key, N);
            if (value != 0xFFFFFFFFu) {
                n[i] = value;
                w[i] = (1.0f - abs(qx - x - 0.5f)) * (1.0f - abs(qy - y - 0.5f)) * (1.0f - abs(qz - z - 0.5f));
                w_sum += w[i];
            }
        }
    }

    if (w_sum < 1e-12f) {
        for (int i = 0; i < 8; i++) {
            neighbor[tid * 8 + i] = n[i];
            weight[tid * 8 + i] = 0.0f;
        }
    } else {
        float inv_w = 1.0f / w_sum;
        for (int i = 0; i < 8; i++) {
            w[i] *= inv_w;
            neighbor[tid * 8 + i] = n[i];
            weight[tid * 8 + i] = w[i];
        }
    }
}

// ============================================================================
// Grid sample trilinear — uint64 keys (split hi/lo)
// ============================================================================

kernel void grid_sample_trilinear_u64(
    const device uint* keys_hi        [[buffer(0)]],
    const device uint* keys_lo        [[buffer(1)]],
    const device uint* hashmap_vals   [[buffer(2)]],
    const device float* query         [[buffer(3)]],
    device uint* neighbor             [[buffer(4)]],
    device float* weight              [[buffer(5)]],
    constant uint& N                  [[buffer(6)]],
    constant uint& B                  [[buffer(7)]],
    constant uint& L                  [[buffer(8)]],
    constant int& W                   [[buffer(9)]],
    constant int& H                   [[buffer(10)]],
    constant int& D                   [[buffer(11)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= B * L) return;
    int b = tid / L;
    float qx = query[3 * tid];
    float qy = query[3 * tid + 1];
    float qz = query[3 * tid + 2];

    uint n[8];
    float w[8];
    float w_sum = 0.0f;

    for (int i = 0; i < 8; i++) {
        n[i] = 0xFFFFFFFFu;
        w[i] = 0.0f;
    }

    int base_x = (int)floor(qx - 0.5f);
    int base_y = (int)floor(qy - 0.5f);
    int base_z = (int)floor(qz - 0.5f);

    for (int i = 0; i < 8; i++) {
        int x = base_x + (i & 1);
        int y = base_y + ((i >> 1) & 1);
        int z = base_z + ((i >> 2) & 1);
        if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
            ulong key = flatten_3d(b, x, y, z, W, H, D);
            uint value = linear_probing_lookup_u64(keys_hi, keys_lo, hashmap_vals, key, N);
            if (value != 0xFFFFFFFFu) {
                n[i] = value;
                w[i] = (1.0f - abs(qx - x - 0.5f)) * (1.0f - abs(qy - y - 0.5f)) * (1.0f - abs(qz - z - 0.5f));
                w_sum += w[i];
            }
        }
    }

    if (w_sum < 1e-12f) {
        for (int i = 0; i < 8; i++) {
            neighbor[tid * 8 + i] = n[i];
            weight[tid * 8 + i] = 0.0f;
        }
    } else {
        float inv_w = 1.0f / w_sum;
        for (int i = 0; i < 8; i++) {
            w[i] *= inv_w;
            neighbor[tid * 8 + i] = n[i];
            weight[tid * 8 + i] = w[i];
        }
    }
}
