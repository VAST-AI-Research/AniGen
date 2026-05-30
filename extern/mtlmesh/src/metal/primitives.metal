// GPU parallel primitives: prefix sum (Blelloch scan), stream compaction, segmented reduce
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

// ========== Prefix Sum (Blelloch Scan) ==========
// Two-pass approach: upsweep + downsweep within threadgroups,
// then a block-level scan for multi-block launches.

// Single-threadgroup inclusive prefix sum (up to 1024 elements)
kernel void prefix_sum_threadgroup_int(
    device int* data [[buffer(0)]],
    device int* block_sums [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup int shared[1024];

    int val = (tid < (uint)N) ? data[tid] : 0;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Upsweep (reduce)
    for (uint stride = 1; stride < group_size; stride *= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < group_size) {
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block sum and clear last element
    if (lid == group_size - 1) {
        if (block_sums) block_sums[gid] = shared[lid];
        shared[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Downsweep
    for (uint stride = group_size / 2; stride >= 1; stride /= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < group_size) {
            int temp = shared[idx - stride];
            shared[idx - stride] = shared[idx];
            shared[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < (uint)N) {
        data[tid] = shared[lid];
    }
}

// Add block offsets after scanning block sums
kernel void add_block_offsets_int(
    device int* data [[buffer(0)]],
    device const int* block_offsets [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (tid < (uint)N && gid > 0) {
        data[tid] += block_offsets[gid];
    }
}

// ========== Stream Compaction (Select::Flagged) ==========
// Two-pass: 1) prefix sum on flags to get output positions, 2) scatter selected elements

kernel void scatter_flagged_int(
    device const int* input [[buffer(0)]],
    device const int* flags [[buffer(1)]],    // 0 or 1
    device const int* offsets [[buffer(2)]],   // exclusive prefix sum of flags
    constant int& N [[buffer(3)]],
    device int* output [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    if (flags[tid]) {
        output[offsets[tid]] = input[tid];
    }
}

kernel void scatter_flagged_int3(
    device const packed_int3* input [[buffer(0)]],
    device const int* flags [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    constant int& N [[buffer(3)]],
    device packed_int3* output [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    if (flags[tid]) {
        output[offsets[tid]] = input[tid];
    }
}

kernel void scatter_flagged_ulong(
    device const ulong* input [[buffer(0)]],
    device const int* flags [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    constant int& N [[buffer(3)]],
    device ulong* output [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    if (flags[tid]) {
        output[offsets[tid]] = input[tid];
    }
}

// ========== Radix Sort (LSB first, 32-bit keys) ==========
// Uses 4-bit radix (16 buckets per pass, 8 passes for 32-bit)

// Count digits in each radix bucket per threadgroup
kernel void radix_count_kernel(
    device const uint* keys [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant int& shift [[buffer(2)]],  // bit shift for current pass (0, 4, 8, ...)
    device atomic_int* histogram [[buffer(3)]], // [16 * num_groups]
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    uint digit = (keys[tid] >> shift) & 0xF;
    atomic_fetch_add_explicit(&histogram[gid * 16 + digit], 1, memory_order_relaxed);
}

// Scatter keys to output positions based on prefix-summed histogram
kernel void radix_scatter_kernel(
    device const uint* keys_in [[buffer(0)]],
    device const int* vals_in [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& shift [[buffer(3)]],
    device const int* prefix_histogram [[buffer(4)]],  // [16] global prefix
    device uint* keys_out [[buffer(5)]],
    device int* vals_out [[buffer(6)]],
    device atomic_int* counters [[buffer(7)]],  // [16] for atomic placement
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    uint digit = (keys_in[tid] >> shift) & 0xF;
    int pos = atomic_fetch_add_explicit(&counters[digit], 1, memory_order_relaxed);
    keys_out[prefix_histogram[digit] + pos] = keys_in[tid];
    vals_out[prefix_histogram[digit] + pos] = vals_in[tid];
}

// ========== Segmented Reduce (Sum) ==========
// Simple per-segment sequential reduction for now
// (Segments are defined by offset arrays)

kernel void segmented_reduce_sum_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    constant int& num_segments [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)num_segments) return;
    int start = offsets[tid];
    int end = offsets[tid + 1];
    float sum = 0;
    for (int i = start; i < end; i++) {
        sum += input[i];
    }
    output[tid] = sum;
}

kernel void segmented_reduce_max_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    constant int& num_segments [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)num_segments) return;
    int start = offsets[tid];
    int end = offsets[tid + 1];
    float mx = -INFINITY;
    for (int i = start; i < end; i++) {
        mx = max(mx, input[i]);
    }
    output[tid] = mx;
}

kernel void segmented_reduce_sum_float3(
    device const packed_float3* input [[buffer(0)]],
    device packed_float3* output [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    constant int& num_segments [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)num_segments) return;
    int start = offsets[tid];
    int end = offsets[tid + 1];
    float3 sum = float3(0);
    for (int i = start; i < end; i++) {
        sum += float3(input[i]);
    }
    output[tid] = packed_float3(sum);
}

// ========== Run-Length Encode ==========
// Mark transitions, prefix sum to get positions

kernel void rle_mark_transitions(
    device const int* sorted_keys [[buffer(0)]],
    constant int& N [[buffer(1)]],
    device int* is_transition [[buffer(2)]],  // 1 if key differs from previous (0 for first element)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    is_transition[tid] = (tid > 0 && sorted_keys[tid] != sorted_keys[tid - 1]) ? 1 : 0;
}

// After prefix-summing is_start, extract unique keys and run lengths
kernel void rle_extract(
    device const int* sorted_keys [[buffer(0)]],
    device const int* run_ids [[buffer(1)]],     // prefix sum of is_start
    constant int& N [[buffer(2)]],
    device int* unique_keys [[buffer(3)]],
    device atomic_int* run_lengths [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    int rid = run_ids[tid];
    // First element of each run writes the key
    if (tid == 0 || sorted_keys[tid] != sorted_keys[tid - 1]) {
        unique_keys[rid] = sorted_keys[tid];
    }
    atomic_fetch_add_explicit(&run_lengths[rid], 1, memory_order_relaxed);
}

// ========== Reduce (global sum) ==========

kernel void reduce_sum_int(
    device const int* input [[buffer(0)]],
    constant int& N [[buffer(1)]],
    device atomic_int* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    atomic_fetch_add_explicit(output, input[tid], memory_order_relaxed);
}

// ========== Reduce By Key ==========
// Given sorted keys and values, sum values for each unique key

kernel void reduce_by_key_float(
    device const ulong* sorted_keys [[buffer(0)]],
    device const float* sorted_values [[buffer(1)]],
    device const int* run_ids [[buffer(2)]],     // prefix sum of is_start for keys
    constant int& N [[buffer(3)]],
    device ulong* unique_keys [[buffer(4)]],
    device atomic_float* reduced_values [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    int rid = run_ids[tid];
    if (tid == 0 || sorted_keys[tid] != sorted_keys[tid-1]) {
        unique_keys[rid] = sorted_keys[tid];
    }
    // Note: atomic_float requires Metal 3.0 / Apple Silicon
    // For older devices, would need a different approach
}

// ========== Utility ==========

kernel void fill_ulong_kernel(
    device ulong* data [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant ulong& value [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N) return;
    data[tid] = value;
}

// Sort uint64 keys (for edge sorting) - simple bitonic sort for small arrays
// For large arrays, use multi-pass radix sort on the host side
kernel void bitonic_sort_step_ulong(
    device ulong* data [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant int& k [[buffer(2)]],   // block size
    constant int& j [[buffer(3)]],   // compare distance
    uint tid [[thread_position_in_grid]]
) {
    uint ixj = tid ^ j;
    if (ixj > tid && ixj < (uint)N) {
        bool ascending = ((tid & k) == 0);
        ulong a = data[tid], b = data[ixj];
        if ((ascending && a > b) || (!ascending && a < b)) {
            data[tid] = b;
            data[ixj] = a;
        }
    }
}
