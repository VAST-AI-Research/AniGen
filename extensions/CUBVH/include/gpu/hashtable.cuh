#pragma once

#include <gpu/common.h>
#include <gpu/gpu_memory.h>

namespace cubvh {

// --- N-dim integer static hash table ---
// This module implements a minimal open-addressed hash table for integer ND coordinates on CUDA.
// Note: this is a STATIC hashtable, only build it once and use it for queries!
//
// Shapes and conventions (all buffers are device pointers with int32 entries unless stated otherwise):
// - coords:   [N * D] layout contiguous as D-tuple per row (default D=3 for 3D).
// - queries:  [M * D] layout contiguous as D-tuple per row.
// - table_kvs: [capacity * 2]; for slot s:
//     table_kvs[2*s + 0] = slot marker (-1 means empty; any other value means occupied)
//     table_kvs[2*s + 1] = row index into coords (0..N-1), or -1 if not set
// - out_indices: [M]; contains row index into coords or -1 if not found.

// Table layout: flattened array of length 2 * capacity (int32)
// - table_kvs[2*slot + 0]: slot marker; -1 means empty; any other value means occupied
// - table_kvs[2*slot + 1]: index into coords (row index)


// --- Hash helper (CityHash-like mix for 32-bit ints) ---
// Used to map a small fixed-length integer key to a slot in [0, capacity).
__device__ inline uint32_t _hash_city32_step(uint32_t hash_val, uint32_t key) {
    hash_val += key * 0x9E3779B9u;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85EBCA6Bu;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xC2B2AE35u;
    hash_val ^= hash_val >> 16;
    return hash_val;
}

struct CityHash {
    // key: pointer to first element; key_dim: number of ints (here 3 in practice)
    // capacity: table capacity (number of slots)
    __device__ inline static int hash(const int* key, int key_dim, int capacity) {
        uint32_t h = 0u;
        for (int i = 0; i < key_dim; ++i) {
            h = _hash_city32_step(h, (uint32_t)key[i]);
        }
        int signed_h = (int)h;
        // ensure non-negative modulo
        int slot = signed_h % capacity;
        if (slot < 0) slot += capacity;
        return slot;
    }
};

__device__ inline bool _vec_equal(const int* a, const int* b, int dim) {
    for (int i = 0; i < dim; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// N-D key search in the hash table (default D=3)
__device__ inline int _search_hash_table(
    const int* __restrict__ table_kvs,
    const int* __restrict__ coords,
    const int* __restrict__ query_key,
    int table_capacity,
    int num_dims
) {
    // query_key: [D]
    // coords: [N * D]
    // table_kvs: [capacity * 2]
    int slot = CityHash::hash(query_key, num_dims, table_capacity);
    const int begin = slot;
    int attempts = 0;
    while (attempts < table_capacity) {
        const int marker = table_kvs[slot * 2 + 0];
        if (marker == -1) {
            return -1; // empty slot encountered => not present
        }
        const int vec_idx = table_kvs[slot * 2 + 1];
        if (vec_idx != -1) {
            const int* candidate = &coords[vec_idx * num_dims];
            if (_vec_equal(candidate, query_key, num_dims)) {
                return vec_idx;
            }
        }
        slot = (slot + 1) % table_capacity;
        if (slot == begin) return -1; // full cycle
        ++attempts;
    }
    return -1;
}

// --- Kernels ---

// Initialize table: set both key marker and value index to -1
// table_kvs: [capacity * 2]
__global__ void prepare_key_value_pairs_kernel(int* table_kvs, int capacity) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < capacity) {
        table_kvs[2 * tid + 0] = -1;
        table_kvs[2 * tid + 1] = -1;
    }
}

// Insert a batch of N-D coords (default D=3)
// table_kvs: [capacity * 2]
// coords:    [num_keys * num_dims]
// num_keys:  N (rows in coords)
__global__ void insert_kernel(
    int* __restrict__ table_kvs,
    const int* __restrict__ coords,
    int num_keys,
    int num_dims,
    int table_capacity
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;
    const int* key = &coords[idx * num_dims];
    int slot = CityHash::hash(key, num_dims, table_capacity);
    const int begin = slot;
    int attempts = 0;
    while (attempts < table_capacity) {
        int* marker_ptr = &table_kvs[slot * 2 + 0];
        const int prev = atomicCAS(marker_ptr, -1, slot);
        if (prev == -1) {
            table_kvs[slot * 2 + 1] = idx; // publish index
            return;
        }
        slot = (slot + 1) % table_capacity;
        if (slot == begin) return; // table full or no free slot
        ++attempts;
    }
}


// Search kernel for arbitrary queries (N-D; default D=3)
// table_kvs:   [capacity * 2]
// coords:      [N * num_dims]
// queries:     [M * num_dims]
// out_indices: [M]
__global__ void search_kernel(
    const int* __restrict__ table_kvs,
    const int* __restrict__ coords,
    const int* __restrict__ queries,
    int num_queries,
    int table_capacity,
    int num_dims,
    int* __restrict__ out_indices
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    const int* q = &queries[idx * num_dims];
    out_indices[idx] = _search_hash_table(table_kvs, coords, q, table_capacity, num_dims);
}


// Lightweight RAII helper for a device hash table backing storage
struct HashTableInt {
    // Device-side storage for the hash table key-value slots.
    // Layout: [capacity * 2] (see top-level description for meaning of each entry)
    GPUMemory<int> table_kvs; // length = 2 * capacity
    int capacity = 0;
    const int* d_coords = nullptr; // external storage of keys [num_coords * num_dims]
    int num_coords = 0;
    int num_dims = 3; // default dimension

    void resize(int new_capacity) {
        capacity = new_capacity;
        table_kvs.resize((size_t)capacity * 2);
    }

    // Set number of dimensions D (default 3)
    void set_num_dims(int d) { num_dims = d; }

    // Initialize/prepare table (set all slots to -1)
    // table_kvs: [capacity * 2]
    void prepare(cudaStream_t stream) {
        if (capacity <= 0) return;
        const uint32_t tpb = 256u;
        const uint32_t blocks = (uint32_t)div_round_up(capacity, (int)tpb);
        prepare_key_value_pairs_kernel<<<blocks, tpb, 0, stream>>>(table_kvs.data(), capacity);
    }

    // Insert a batch of coords
    // d_coords_in: [n_keys * num_dims]
    void insert(const int* d_coords_in, int n_keys, cudaStream_t stream) {
        d_coords = d_coords_in;
        num_coords = n_keys;
        if (capacity <= 0 || n_keys <= 0) return;
        const uint32_t tpb = 256u;
        const uint32_t blocks = (uint32_t)div_round_up(n_keys, (int)tpb);
        insert_kernel<<<blocks, tpb, 0, stream>>>(table_kvs.data(), d_coords, n_keys, num_dims, capacity);
    }


    // Build convenience: set capacity, prepare, then insert in one call
    // d_coords_in: [n_keys * num_dims]
    void build(const int* d_coords_in, int n_keys, cudaStream_t stream) {
        // initialize capacity = max(16, 2 * n_keys)
        int desired_capacity = n_keys * 2;
        if (desired_capacity < 16) desired_capacity = 16;
        resize(desired_capacity);
        prepare(stream);
        insert(d_coords_in, n_keys, stream);
    }

    // Search a batch of queries; writes index or -1 per query
    // d_queries:     [n_queries * num_dims]
    // d_out_indices: [n_queries]
    void search(const int* d_queries, int n_queries, int* d_out_indices, cudaStream_t stream) const {
        if (capacity <= 0 || n_queries <= 0) return;
        const uint32_t tpb = 256u;
        const uint32_t blocks = (uint32_t)div_round_up(n_queries, (int)tpb);
        search_kernel<<<blocks, tpb, 0, stream>>>(table_kvs.data(), d_coords, d_queries, n_queries, capacity, num_dims, d_out_indices);
    }
};

} // namespace cubvh