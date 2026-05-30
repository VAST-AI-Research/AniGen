#ifndef HASH_H
#define HASH_H

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Murmur3 hash
// ============================================================================

inline uint murmur3_32(uint k, uint N) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}

inline ulong murmur3_64(ulong k, uint N) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k % N;
}

// ============================================================================
// Linear probing — uint32 keys
// ============================================================================

inline void linear_probing_insert_u32(
    device atomic_uint* hashmap_keys,
    device uint* hashmap_values,
    uint key,
    uint value,
    uint N
) {
    uint slot = murmur3_32(key, N);
    while (true) {
        uint expected = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(
                &hashmap_keys[slot], &expected, key,
                memory_order_relaxed, memory_order_relaxed)) {
            hashmap_values[slot] = value;
            return;
        }
        if (expected == key) {
            hashmap_values[slot] = value;
            return;
        }
        slot = (slot + 1) % N;
    }
}

inline uint linear_probing_lookup_u32(
    const device uint* hashmap_keys,
    const device uint* hashmap_values,
    uint key,
    uint N
) {
    uint slot = murmur3_32(key, N);
    while (true) {
        uint k = hashmap_keys[slot];
        if (k == 0xFFFFFFFFu) {
            return 0xFFFFFFFFu;
        }
        if (k == key) {
            return hashmap_values[slot];
        }
        slot = (slot + 1) % N;
    }
}

// ============================================================================
// Linear probing — uint64 keys via split (hi, lo) atomic_uint pairs
// Two uint32 slots per key: keys_hi[slot] = upper 32, keys_lo[slot] = lower 32.
// CAS on hi first, then lo. Race-safe: if hi CAS fails, another thread owns slot.
// ============================================================================

inline void linear_probing_insert_u64(
    device atomic_uint* keys_hi,
    device atomic_uint* keys_lo,
    device uint* hashmap_values,
    ulong key,
    uint value,
    uint N
) {
    uint hi = (uint)(key >> 32);
    uint lo = (uint)(key & 0xFFFFFFFFULL);
    uint slot = (uint)murmur3_64(key, N);

    while (true) {
        // Try to claim the slot by CAS on hi part
        uint exp_hi = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(
                &keys_hi[slot], &exp_hi, hi,
                memory_order_relaxed, memory_order_relaxed)) {
            // Won the slot — write lo and value
            atomic_store_explicit(&keys_lo[slot], lo, memory_order_relaxed);
            hashmap_values[slot] = value;
            return;
        }
        // Slot already taken — check if it's our key
        if (exp_hi == hi) {
            uint cur_lo = atomic_load_explicit(&keys_lo[slot], memory_order_relaxed);
            if (cur_lo == lo) {
                hashmap_values[slot] = value;
                return;
            }
        }
        slot = (slot + 1) % N;
    }
}

inline uint linear_probing_lookup_u64(
    const device uint* keys_hi,
    const device uint* keys_lo,
    const device uint* hashmap_values,
    ulong key,
    uint N
) {
    uint hi = (uint)(key >> 32);
    uint lo = (uint)(key & 0xFFFFFFFFULL);
    uint slot = (uint)murmur3_64(key, N);

    while (true) {
        uint kh = keys_hi[slot];
        if (kh == 0xFFFFFFFFu) {
            return 0xFFFFFFFFu;  // empty slot
        }
        if (kh == hi) {
            uint kl = keys_lo[slot];
            if (kl == lo) {
                return hashmap_values[slot];
            }
        }
        slot = (slot + 1) % N;
    }
}

// ============================================================================
// Linear probing — uint64 keys, packed layout.
//
// The split (hi, lo) path above exists for CPU-mode dispatch, where PyTorch
// u64 tensors are decomposed into two u32 tensors via torch::from_blob so
// the resulting (hi, lo) buffers are stable Metal buffers. On MPS the split
// requires copies (slice is a strided view; it must be `.contiguous()`d,
// which allocates).
//
// The packed variant below takes a single buffer that views the u64 hashmap
// as interleaved u32 pairs [lo0, hi0, lo1, hi1, ...] (PyTorch's u64 layout
// on Apple Silicon, little-endian). CAS on u32 is fully supported; Metal's
// 64-bit atomic CAS is not. Same algorithm as the (hi, lo) split: CAS the
// hi half of the slot to claim it, then write lo + value. Zero-copy on MPS
// because we reinterpret the u64 tensor's underlying MTLBuffer as a u32
// stream — the backing memory is identical.
// ============================================================================

inline void linear_probing_insert_u64_packed(
    device atomic_uint* packed_keys,   // interleaved [lo0, hi0, lo1, hi1, ...]
    device uint* hashmap_values,
    ulong key,
    uint value,
    uint N
) {
    uint hi = (uint)(key >> 32);
    uint lo = (uint)(key & 0xFFFFFFFFULL);
    uint slot = (uint)murmur3_64(key, N);

    while (true) {
        uint hi_idx = slot * 2 + 1;
        uint lo_idx = slot * 2;
        uint exp_hi = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(
                &packed_keys[hi_idx], &exp_hi, hi,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&packed_keys[lo_idx], lo, memory_order_relaxed);
            hashmap_values[slot] = value;
            return;
        }
        if (exp_hi == hi) {
            uint cur_lo = atomic_load_explicit(&packed_keys[lo_idx], memory_order_relaxed);
            if (cur_lo == lo) {
                hashmap_values[slot] = value;
                return;
            }
        }
        slot = (slot + 1) % N;
    }
}

inline uint linear_probing_lookup_u64_packed(
    const device uint* packed_keys,   // interleaved [lo0, hi0, lo1, hi1, ...]
    const device uint* hashmap_values,
    ulong key,
    uint N
) {
    uint hi = (uint)(key >> 32);
    uint lo = (uint)(key & 0xFFFFFFFFULL);
    uint slot = (uint)murmur3_64(key, N);

    while (true) {
        uint kh = packed_keys[slot * 2 + 1];
        if (kh == 0xFFFFFFFFu) {
            return 0xFFFFFFFFu;
        }
        if (kh == hi) {
            uint kl = packed_keys[slot * 2];
            if (kl == lo) {
                return hashmap_values[slot];
            }
        }
        slot = (slot + 1) % N;
    }
}

// ============================================================================
// 3D coordinate flattening
// ============================================================================

inline ulong flatten_3d(int b, int x, int y, int z, int W, int H, int D) {
    return (ulong)b * W * H * D + (ulong)x * H * D + (ulong)y * D + z;
}

#endif // HASH_H
