// Hash table compute kernels — Murmur3 hashing with linear probing
#include <metal_stdlib>
using namespace metal;


// ──────────────────────────────────────────────────────────────────────
// Murmur3 hash functions
// ──────────────────────────────────────────────────────────────────────

inline uint hash32(uint k, uint N) {
    if (N == 0) return 0;  // guard div-by-zero (caller loops are bounded by N)
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}

inline uint hash64(ulong k, uint N) {
    if (N == 0) return 0;  // guard div-by-zero (caller loops are bounded by N)
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return uint(k % ulong(N));
}


// ──────────────────────────────────────────────────────────────────────
// Linear probing — uint32 keys
// ──────────────────────────────────────────────────────────────────────

inline void linear_probing_insert_u32_u32(
    device atomic_uint* hashmap_keys,
    device uint* hashmap_values,
    uint key, uint value, uint N
) {
    uint slot = hash32(key, N);
    // MSL only has weak CAS, which may fail SPURIOUSLY on an empty slot. If we
    // advanced the probe on a spurious failure we would leave a gap in the chain
    // and lookups (which stop at the first empty slot) would miss the key. So:
    // only advance when the slot is genuinely occupied by a different key; retry
    // the SAME slot on a spurious failure. Total iters are capped (no GPU spin).
    uint advanced = 0;
    for (uint iter = 0; iter < 4u * N + 64u; ++iter) {
        uint expected = 0xFFFFFFFF;
        bool ok = atomic_compare_exchange_weak_explicit(
            &hashmap_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed);
        if (ok || expected == key) {
            hashmap_values[slot] = value;
            return;
        }
        if (expected == 0xFFFFFFFF) continue;  // spurious failure — retry same slot
        slot = (slot + 1 < N) ? slot + 1 : 0;
        if (++advanced >= N) return;           // table genuinely full
    }
}

inline void linear_probing_insert_u32_u64(
    device atomic_uint* hashmap_keys,
    device ulong* hashmap_values,
    uint key, ulong value, uint N
) {
    uint slot = hash32(key, N);
    // MSL only has weak CAS, which may fail SPURIOUSLY on an empty slot. If we
    // advanced the probe on a spurious failure we would leave a gap in the chain
    // and lookups (which stop at the first empty slot) would miss the key. So:
    // only advance when the slot is genuinely occupied by a different key; retry
    // the SAME slot on a spurious failure. Total iters are capped (no GPU spin).
    uint advanced = 0;
    for (uint iter = 0; iter < 4u * N + 64u; ++iter) {
        uint expected = 0xFFFFFFFF;
        bool ok = atomic_compare_exchange_weak_explicit(
            &hashmap_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed);
        if (ok || expected == key) {
            hashmap_values[slot] = value;
            return;
        }
        if (expected == 0xFFFFFFFF) continue;  // spurious failure — retry same slot
        slot = (slot + 1 < N) ? slot + 1 : 0;
        if (++advanced >= N) return;           // table genuinely full
    }
}

inline uint linear_probing_lookup_u32_u32(
    const device uint* hashmap_keys,
    const device uint* hashmap_values,
    uint key, uint N
) {
    uint slot = hash32(key, N);
    for (uint probes = 0; probes < N; ++probes) {
        uint prev = hashmap_keys[slot];
        if (prev == 0xFFFFFFFF) return 0xFFFFFFFF;
        if (prev == key) return hashmap_values[slot];
        slot = (slot + 1 < N) ? slot + 1 : 0;
    }
    return 0xFFFFFFFF;  // table full / not found — never spin the GPU
}

inline ulong linear_probing_lookup_u32_u64(
    const device uint* hashmap_keys,
    const device ulong* hashmap_values,
    uint key, uint N
) {
    uint slot = hash32(key, N);
    for (uint probes = 0; probes < N; ++probes) {
        uint prev = hashmap_keys[slot];
        if (prev == 0xFFFFFFFF) return 0xFFFFFFFFFFFFFFFFULL;
        if (prev == key) return hashmap_values[slot];
        slot = (slot + 1 < N) ? slot + 1 : 0;
    }
    return 0xFFFFFFFFFFFFFFFFULL;  // table full / not found — never spin the GPU
}


// ──────────────────────────────────────────────────────────────────────
// Linear probing — uint64 keys (split hi/lo)
// 64-bit keys stored as pairs of atomic_uint: slot*2 = high, slot*2+1 = low.
// Insert: CAS on high word, then write low word + value.
// Lookup: read both halves, compare as combined 64-bit key.
// ──────────────────────────────────────────────────────────────────────

inline void linear_probing_insert_u64_u32(
    device atomic_uint* hashmap_keys_split,  // [2*N] — pairs of (hi, lo)
    device uint* hashmap_values,
    ulong key, uint value, uint N
) {
    uint key_hi = uint(key >> 32);
    uint key_lo = uint(key & 0xFFFFFFFFu);
    uint slot = hash64(key, N);
    // See u32 insert: weak CAS may fail spuriously; retry the same slot then.
    uint advanced = 0;
    for (uint iter = 0; iter < 4u * N + 64u; ++iter) {
        uint expected_hi = 0xFFFFFFFF;
        bool ok = atomic_compare_exchange_weak_explicit(
            &hashmap_keys_split[slot * 2], &expected_hi, key_hi,
            memory_order_relaxed, memory_order_relaxed);
        if (ok) {
            // Won the slot — write low half and value
            atomic_store_explicit(&hashmap_keys_split[slot * 2 + 1], key_lo, memory_order_relaxed);
            hashmap_values[slot] = value;
            return;
        }
        if (expected_hi == 0xFFFFFFFF) continue;  // spurious failure — retry same slot
        if (expected_hi == key_hi) {
            uint existing_lo = atomic_load_explicit(&hashmap_keys_split[slot * 2 + 1], memory_order_relaxed);
            if (existing_lo == key_lo) {
                // Key already exists — update value
                hashmap_values[slot] = value;
                return;
            }
        }
        slot = (slot + 1 < N) ? slot + 1 : 0;
        if (++advanced >= N) return;              // table genuinely full
    }
}

inline void linear_probing_insert_u64_u64(
    device atomic_uint* hashmap_keys_split,  // [2*N]
    device ulong* hashmap_values,
    ulong key, ulong value, uint N
) {
    uint key_hi = uint(key >> 32);
    uint key_lo = uint(key & 0xFFFFFFFFu);
    uint slot = hash64(key, N);
    // See u32 insert: weak CAS may fail spuriously; retry the same slot then.
    uint advanced = 0;
    for (uint iter = 0; iter < 4u * N + 64u; ++iter) {
        uint expected_hi = 0xFFFFFFFF;
        bool ok = atomic_compare_exchange_weak_explicit(
            &hashmap_keys_split[slot * 2], &expected_hi, key_hi,
            memory_order_relaxed, memory_order_relaxed);
        if (ok) {
            atomic_store_explicit(&hashmap_keys_split[slot * 2 + 1], key_lo, memory_order_relaxed);
            hashmap_values[slot] = value;
            return;
        }
        if (expected_hi == 0xFFFFFFFF) continue;  // spurious failure — retry same slot
        if (expected_hi == key_hi) {
            uint existing_lo = atomic_load_explicit(&hashmap_keys_split[slot * 2 + 1], memory_order_relaxed);
            if (existing_lo == key_lo) {
                hashmap_values[slot] = value;
                return;
            }
        }
        slot = (slot + 1 < N) ? slot + 1 : 0;
        if (++advanced >= N) return;              // table genuinely full
    }
}

inline uint linear_probing_lookup_u64_u32(
    const device uint* hashmap_keys_split,  // [2*N] — pairs of (hi, lo)
    const device uint* hashmap_values,
    ulong key, uint N
) {
    uint key_hi = uint(key >> 32);
    uint key_lo = uint(key & 0xFFFFFFFFu);
    uint slot = hash64(key, N);
    for (uint probes = 0; probes < N; ++probes) {
        uint prev_hi = hashmap_keys_split[slot * 2];
        if (prev_hi == 0xFFFFFFFF) return 0xFFFFFFFF;  // empty slot
        uint prev_lo = hashmap_keys_split[slot * 2 + 1];
        if (prev_hi == key_hi && prev_lo == key_lo) return hashmap_values[slot];
        slot = (slot + 1 < N) ? slot + 1 : 0;
    }
    return 0xFFFFFFFF;  // table full / not found — never spin the GPU
}

inline ulong linear_probing_lookup_u64_u64(
    const device uint* hashmap_keys_split,
    const device ulong* hashmap_values,
    ulong key, uint N
) {
    uint key_hi = uint(key >> 32);
    uint key_lo = uint(key & 0xFFFFFFFFu);
    uint slot = hash64(key, N);
    for (uint probes = 0; probes < N; ++probes) {
        uint prev_hi = hashmap_keys_split[slot * 2];
        if (prev_hi == 0xFFFFFFFF) return 0xFFFFFFFFFFFFFFFFULL;
        uint prev_lo = hashmap_keys_split[slot * 2 + 1];
        if (prev_hi == key_hi && prev_lo == key_lo) return hashmap_values[slot];
        slot = (slot + 1 < N) ? slot + 1 : 0;
    }
    return 0xFFFFFFFFFFFFFFFFULL;  // table full / not found — never spin the GPU
}


// ──────────────────────────────────────────────────────────────────────
// hashmap_insert — 1D keys
// ──────────────────────────────────────────────────────────────────────

kernel void hashmap_insert_u32_u32_kernel(
    device atomic_uint* hashmap_keys  [[buffer(0)]],
    device uint*        hashmap_values [[buffer(1)]],
    const device uint*  keys           [[buffer(2)]],
    const device uint*  values         [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    linear_probing_insert_u32_u32(hashmap_keys, hashmap_values, keys[tid], values[tid], N);
}

kernel void hashmap_insert_u32_u64_kernel(
    device atomic_uint* hashmap_keys  [[buffer(0)]],
    device ulong*       hashmap_values [[buffer(1)]],
    const device uint*  keys           [[buffer(2)]],
    const device ulong* values         [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    linear_probing_insert_u32_u64(hashmap_keys, hashmap_values, keys[tid], values[tid], N);
}

kernel void hashmap_insert_u64_u32_kernel(
    device atomic_uint* hashmap_keys  [[buffer(0)]],  // split u64: [2*N] pairs of (hi, lo)
    device uint*         hashmap_values [[buffer(1)]],
    const device ulong*  keys           [[buffer(2)]],
    const device uint*   values         [[buffer(3)]],
    constant uint&       N              [[buffer(4)]],
    constant uint&       M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    linear_probing_insert_u64_u32(hashmap_keys, hashmap_values, keys[tid], values[tid], N);
}

kernel void hashmap_insert_u64_u64_kernel(
    device atomic_uint* hashmap_keys  [[buffer(0)]],  // split u64: [2*N] pairs of (hi, lo)
    device ulong*        hashmap_values [[buffer(1)]],
    const device ulong*  keys           [[buffer(2)]],
    const device ulong*  values         [[buffer(3)]],
    constant uint&       N              [[buffer(4)]],
    constant uint&       M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    linear_probing_insert_u64_u64(hashmap_keys, hashmap_values, keys[tid], values[tid], N);
}


// ──────────────────────────────────────────────────────────────────────
// hashmap_lookup — 1D keys
// ──────────────────────────────────────────────────────────────────────

kernel void hashmap_lookup_u32_u32_kernel(
    const device uint*  hashmap_keys   [[buffer(0)]],
    const device uint*  hashmap_values [[buffer(1)]],
    const device uint*  keys           [[buffer(2)]],
    device uint*        out_values     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    out_values[tid] = linear_probing_lookup_u32_u32(hashmap_keys, hashmap_values, keys[tid], N);
}

kernel void hashmap_lookup_u32_u64_kernel(
    const device uint*  hashmap_keys   [[buffer(0)]],
    const device ulong* hashmap_values [[buffer(1)]],
    const device uint*  keys           [[buffer(2)]],
    device ulong*       out_values     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    out_values[tid] = linear_probing_lookup_u32_u64(hashmap_keys, hashmap_values, keys[tid], N);
}

kernel void hashmap_lookup_u64_u32_kernel(
    const device uint* hashmap_keys   [[buffer(0)]],  // split u64: [2*N]
    const device uint*  hashmap_values [[buffer(1)]],
    const device ulong* keys           [[buffer(2)]],
    device uint*        out_values     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    out_values[tid] = linear_probing_lookup_u64_u32(hashmap_keys, hashmap_values, keys[tid], N);
}

kernel void hashmap_lookup_u64_u64_kernel(
    const device uint* hashmap_keys   [[buffer(0)]],  // split u64: [2*N]
    const device ulong* hashmap_values [[buffer(1)]],
    const device ulong* keys           [[buffer(2)]],
    device ulong*       out_values     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      M              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    out_values[tid] = linear_probing_lookup_u64_u64(hashmap_keys, hashmap_values, keys[tid], N);
}


// ──────────────────────────────────────────────────────────────────────
// hashmap_insert_3d — 3D coords packed to flat key
// coords layout: [M, 4] int32 — (batch, x, y, z)
// ──────────────────────────────────────────────────────────────────────

kernel void hashmap_insert_3d_u32_u32_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],
    device uint*            hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    const device uint*      values         [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    uint flat_idx = uint(ulong(b) * ulong(W) * ulong(H) * ulong(D)
                      + ulong(x) * ulong(H) * ulong(D)
                      + ulong(y) * ulong(D) + ulong(z));
    linear_probing_insert_u32_u32(hashmap_keys, hashmap_values, flat_idx, values[tid], N);
}

kernel void hashmap_insert_3d_u32_u64_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],
    device ulong*           hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    const device ulong*     values         [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    uint flat_idx = uint(ulong(b) * ulong(W) * ulong(H) * ulong(D)
                      + ulong(x) * ulong(H) * ulong(D)
                      + ulong(y) * ulong(D) + ulong(z));
    linear_probing_insert_u32_u64(hashmap_keys, hashmap_values, flat_idx, values[tid], N);
}

kernel void hashmap_insert_3d_u64_u32_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],  // split u64: [2*N] pairs of (hi, lo)
    device uint*            hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    const device uint*      values         [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    ulong flat_idx = ulong(b) * ulong(W) * ulong(H) * ulong(D)
                   + ulong(x) * ulong(H) * ulong(D)
                   + ulong(y) * ulong(D) + ulong(z);
    linear_probing_insert_u64_u32(hashmap_keys, hashmap_values, flat_idx, values[tid], N);
}

kernel void hashmap_insert_3d_u64_u64_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],  // split u64: [2*N] pairs of (hi, lo)
    device ulong*           hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    const device ulong*     values         [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    ulong flat_idx = ulong(b) * ulong(W) * ulong(H) * ulong(D)
                   + ulong(x) * ulong(H) * ulong(D)
                   + ulong(y) * ulong(D) + ulong(z);
    linear_probing_insert_u64_u64(hashmap_keys, hashmap_values, flat_idx, values[tid], N);
}


// ──────────────────────────────────────────────────────────────────────
// hashmap_lookup_3d — 3D coords packed to flat key
// ──────────────────────────────────────────────────────────────────────

kernel void hashmap_lookup_3d_u32_u32_kernel(
    const device uint*      hashmap_keys   [[buffer(0)]],
    const device uint*      hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    device uint*            out_values     [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        out_values[tid] = 0xFFFFFFFF;
        return;
    }
    uint flat_idx = uint(ulong(b) * ulong(W) * ulong(H) * ulong(D)
                      + ulong(x) * ulong(H) * ulong(D)
                      + ulong(y) * ulong(D) + ulong(z));
    out_values[tid] = linear_probing_lookup_u32_u32(hashmap_keys, hashmap_values, flat_idx, N);
}

kernel void hashmap_lookup_3d_u32_u64_kernel(
    const device uint*      hashmap_keys   [[buffer(0)]],
    const device ulong*     hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    device ulong*           out_values     [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        out_values[tid] = 0xFFFFFFFFFFFFFFFFULL;
        return;
    }
    uint flat_idx = uint(ulong(b) * ulong(W) * ulong(H) * ulong(D)
                      + ulong(x) * ulong(H) * ulong(D)
                      + ulong(y) * ulong(D) + ulong(z));
    out_values[tid] = linear_probing_lookup_u32_u64(hashmap_keys, hashmap_values, flat_idx, N);
}

kernel void hashmap_lookup_3d_u64_u32_kernel(
    const device uint*      hashmap_keys   [[buffer(0)]],  // split u64: [2*N]
    const device uint*      hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    device uint*            out_values     [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        out_values[tid] = 0xFFFFFFFF;
        return;
    }
    ulong flat_idx = ulong(b) * ulong(W) * ulong(H) * ulong(D)
                   + ulong(x) * ulong(H) * ulong(D)
                   + ulong(y) * ulong(D) + ulong(z);
    out_values[tid] = linear_probing_lookup_u64_u32(hashmap_keys, hashmap_values, flat_idx, N);
}

kernel void hashmap_lookup_3d_u64_u64_kernel(
    const device uint*      hashmap_keys   [[buffer(0)]],  // split u64: [2*N]
    const device ulong*     hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    device ulong*           out_values     [[buffer(3)]],
    constant uint&          N              [[buffer(4)]],
    constant uint&          M              [[buffer(5)]],
    constant int&           W              [[buffer(6)]],
    constant int&           H              [[buffer(7)]],
    constant int&           D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        out_values[tid] = 0xFFFFFFFFFFFFFFFFULL;
        return;
    }
    ulong flat_idx = ulong(b) * ulong(W) * ulong(H) * ulong(D)
                   + ulong(x) * ulong(H) * ulong(D)
                   + ulong(y) * ulong(D) + ulong(z);
    out_values[tid] = linear_probing_lookup_u64_u64(hashmap_keys, hashmap_values, flat_idx, N);
}


// ──────────────────────────────────────────────────────────────────────
// hashmap_insert_3d_idx_as_val — value = thread index
// ──────────────────────────────────────────────────────────────────────

kernel void hashmap_insert_3d_idx_as_val_u32_u32_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],
    device uint*            hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    constant uint&          N              [[buffer(3)]],
    constant uint&          M              [[buffer(4)]],
    constant int&           W              [[buffer(5)]],
    constant int&           H              [[buffer(6)]],
    constant int&           D              [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    uint flat_idx = uint(ulong(b) * ulong(W) * ulong(H) * ulong(D)
                      + ulong(x) * ulong(H) * ulong(D)
                      + ulong(y) * ulong(D) + ulong(z));
    linear_probing_insert_u32_u32(hashmap_keys, hashmap_values, flat_idx, tid, N);
}

kernel void hashmap_insert_3d_idx_as_val_u32_u64_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],
    device ulong*           hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    constant uint&          N              [[buffer(3)]],
    constant uint&          M              [[buffer(4)]],
    constant int&           W              [[buffer(5)]],
    constant int&           H              [[buffer(6)]],
    constant int&           D              [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    uint flat_idx = uint(ulong(b) * ulong(W) * ulong(H) * ulong(D)
                      + ulong(x) * ulong(H) * ulong(D)
                      + ulong(y) * ulong(D) + ulong(z));
    linear_probing_insert_u32_u64(hashmap_keys, hashmap_values, flat_idx, ulong(tid), N);
}

kernel void hashmap_insert_3d_idx_as_val_u64_u32_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],  // split u64: [2*N] pairs of (hi, lo)
    device uint*            hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    constant uint&          N              [[buffer(3)]],
    constant uint&          M              [[buffer(4)]],
    constant int&           W              [[buffer(5)]],
    constant int&           H              [[buffer(6)]],
    constant int&           D              [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    ulong flat_idx = ulong(b) * ulong(W) * ulong(H) * ulong(D)
                   + ulong(x) * ulong(H) * ulong(D)
                   + ulong(y) * ulong(D) + ulong(z);
    linear_probing_insert_u64_u32(hashmap_keys, hashmap_values, flat_idx, tid, N);
}

kernel void hashmap_insert_3d_idx_as_val_u64_u64_kernel(
    device atomic_uint*     hashmap_keys   [[buffer(0)]],  // split u64: [2*N] pairs of (hi, lo)
    device ulong*           hashmap_values [[buffer(1)]],
    const device int4*      coords         [[buffer(2)]],
    constant uint&          N              [[buffer(3)]],
    constant uint&          M              [[buffer(4)]],
    constant int&           W              [[buffer(5)]],
    constant int&           H              [[buffer(6)]],
    constant int&           D              [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;
    int4 coord = coords[tid];
    int b = coord.x, x = coord.y, y = coord.z, z = coord.w;
    ulong flat_idx = ulong(b) * ulong(W) * ulong(H) * ulong(D)
                   + ulong(x) * ulong(H) * ulong(D)
                   + ulong(y) * ulong(D) + ulong(z);
    linear_probing_insert_u64_u64(hashmap_keys, hashmap_values, flat_idx, ulong(tid), N);
}
