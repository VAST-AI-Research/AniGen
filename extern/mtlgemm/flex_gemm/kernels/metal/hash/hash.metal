#include <metal_stdlib>
using namespace metal;

#include "hash.h"

// ============================================================================
// Hashmap insert — generic keys (uint32)
// ============================================================================

kernel void hashmap_insert_u32(
    device atomic_uint* hashmap_keys  [[buffer(0)]],
    device uint* hashmap_values       [[buffer(1)]],
    const device uint* keys           [[buffer(2)]],
    const device uint* values         [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    linear_probing_insert_u32(hashmap_keys, hashmap_values, keys[tid], values[tid], N);
}

// ============================================================================
// Hashmap lookup — generic keys (uint32 keys, uint32 values)
// ============================================================================

kernel void hashmap_lookup_u32_u32(
    const device uint* hashmap_keys   [[buffer(0)]],
    const device uint* hashmap_values [[buffer(1)]],
    const device uint* keys           [[buffer(2)]],
    device uint* values               [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    values[tid] = linear_probing_lookup_u32(hashmap_keys, hashmap_values, keys[tid], N);
}

// ============================================================================
// Hashmap insert 3D — uint32 keys
// ============================================================================

kernel void hashmap_insert_3d_u32(
    device atomic_uint* hashmap_keys  [[buffer(0)]],
    device uint* hashmap_values       [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    const device uint* values         [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    uint key = (uint)flatten_3d(b, x, y, z, W, H, D);
    linear_probing_insert_u32(hashmap_keys, hashmap_values, key, values[tid], N);
}

// ============================================================================
// Hashmap insert 3D — uint64 keys (split hi/lo)
// ============================================================================

kernel void hashmap_insert_3d_u64(
    device atomic_uint* keys_hi       [[buffer(0)]],
    device atomic_uint* keys_lo       [[buffer(1)]],
    device uint* hashmap_values       [[buffer(2)]],
    const device int* coords          [[buffer(3)]],
    const device uint* values         [[buffer(4)]],
    constant uint& N                  [[buffer(5)]],
    constant uint& M                  [[buffer(6)]],
    constant int& W                   [[buffer(7)]],
    constant int& H                   [[buffer(8)]],
    constant int& D                   [[buffer(9)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    ulong key = flatten_3d(b, x, y, z, W, H, D);
    linear_probing_insert_u64(keys_hi, keys_lo, hashmap_values, key, values[tid], N);
}

// ============================================================================
// Hashmap lookup 3D — uint64 keys (split hi/lo)
// ============================================================================

kernel void hashmap_lookup_3d_u64(
    const device uint* keys_hi        [[buffer(0)]],
    const device uint* keys_lo        [[buffer(1)]],
    const device uint* hashmap_values [[buffer(2)]],
    const device int* coords          [[buffer(3)]],
    device uint* values               [[buffer(4)]],
    constant uint& N                  [[buffer(5)]],
    constant uint& M                  [[buffer(6)]],
    constant int& W                   [[buffer(7)]],
    constant int& H                   [[buffer(8)]],
    constant int& D                   [[buffer(9)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        values[tid] = 0xFFFFFFFFu;
        return;
    }
    ulong key = flatten_3d(b, x, y, z, W, H, D);
    values[tid] = linear_probing_lookup_u64(keys_hi, keys_lo, hashmap_values, key, N);
}

// ============================================================================
// Hashmap insert 3D idx-as-val — uint64 keys (split hi/lo)
// ============================================================================

kernel void hashmap_insert_3d_idx_as_val_u64(
    device atomic_uint* keys_hi       [[buffer(0)]],
    device atomic_uint* keys_lo       [[buffer(1)]],
    device uint* hashmap_values       [[buffer(2)]],
    const device int* coords          [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    ulong key = flatten_3d(b, x, y, z, W, H, D);
    linear_probing_insert_u64(keys_hi, keys_lo, hashmap_values, key, tid, N);
}

// ============================================================================
// Hashmap lookup 3D — uint32 keys
// ============================================================================

kernel void hashmap_lookup_3d_u32(
    const device uint* hashmap_keys   [[buffer(0)]],
    const device uint* hashmap_values [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    device uint* values               [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        values[tid] = 0xFFFFFFFFu;
        return;
    }
    uint key = (uint)flatten_3d(b, x, y, z, W, H, D);
    values[tid] = linear_probing_lookup_u32(hashmap_keys, hashmap_values, key, N);
}

// ============================================================================
// Direct u64 variants — take the u64 hashmap buffer zero-copy (MPS path).
// See hash.h for the rationale versus the split (hi, lo) variants above.
// ============================================================================

kernel void hashmap_insert_3d_u64_packed(
    device atomic_uint* packed_keys   [[buffer(0)]],
    device uint* hashmap_values       [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    const device uint* values         [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    ulong key = flatten_3d(b, x, y, z, W, H, D);
    linear_probing_insert_u64_packed(packed_keys, hashmap_values, key, values[tid], N);
}

kernel void hashmap_lookup_3d_u64_packed(
    const device uint* packed_keys    [[buffer(0)]],
    const device uint* hashmap_values [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    device uint* values               [[buffer(3)]],
    constant uint& N                  [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant int& W                   [[buffer(6)]],
    constant int& H                   [[buffer(7)]],
    constant int& D                   [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        values[tid] = 0xFFFFFFFFu;
        return;
    }
    ulong key = flatten_3d(b, x, y, z, W, H, D);
    values[tid] = linear_probing_lookup_u64_packed(packed_keys, hashmap_values, key, N);
}

kernel void hashmap_insert_3d_idx_as_val_u64_packed(
    device atomic_uint* packed_keys   [[buffer(0)]],
    device uint* hashmap_values       [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    constant uint& N                  [[buffer(3)]],
    constant uint& M                  [[buffer(4)]],
    constant int& W                   [[buffer(5)]],
    constant int& H                   [[buffer(6)]],
    constant int& D                   [[buffer(7)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    ulong key = flatten_3d(b, x, y, z, W, H, D);
    linear_probing_insert_u64_packed(packed_keys, hashmap_values, key, tid, N);
}

// ============================================================================
// Hashmap insert 3D idx-as-val — uint32 keys
// ============================================================================

kernel void hashmap_insert_3d_idx_as_val_u32(
    device atomic_uint* hashmap_keys  [[buffer(0)]],
    device uint* hashmap_values       [[buffer(1)]],
    const device int* coords          [[buffer(2)]],
    constant uint& N                  [[buffer(3)]],
    constant uint& M                  [[buffer(4)]],
    constant int& W                   [[buffer(5)]],
    constant int& H                   [[buffer(6)]],
    constant int& D                   [[buffer(7)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int b = coords[tid * 4 + 0];
    int x = coords[tid * 4 + 1];
    int y = coords[tid * 4 + 2];
    int z = coords[tid * 4 + 3];
    uint key = (uint)flatten_3d(b, x, y, z, W, H, D);
    linear_probing_insert_u32(hashmap_keys, hashmap_values, key, tid, N);
}

