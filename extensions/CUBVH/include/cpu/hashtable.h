#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

namespace cubvh {

// --- N-dim integer static hash table (CPU) ---
// This mirrors the CUDA version in include/gpu/hashtable.cuh, but operates on host memory.
// Notes:
// - Static, open-addressed table with linear probing.
// - Keys are rows of int32 of length num_dims in a contiguous buffer.
// - Table storage layout: table_kvs[2 * capacity]
//     table_kvs[2*s + 0] = slot marker (-1 empty; any other value => occupied)
//     table_kvs[2*s + 1] = row index into coords (0..N-1) or -1

// --- Hash helper (CityHash-like mix for 32-bit ints) ---
inline uint32_t _hash_city32_step_cpu(uint32_t hash_val, uint32_t key) {
    hash_val += key * 0x9E3779B9u;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85EBCA6Bu;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xC2B2AE35u;
    hash_val ^= hash_val >> 16;
    return hash_val;
}

struct CityHashCPU {
    inline static int hash(const int* key, int key_dim, int capacity) {
        uint32_t h = 0u;
        for (int i = 0; i < key_dim; ++i) {
            h = _hash_city32_step_cpu(h, static_cast<uint32_t>(key[i]));
        }
        int signed_h = static_cast<int>(h);
        int slot = signed_h % capacity;
        if (slot < 0) slot += capacity;
        return slot;
    }
};

inline bool _vec_equal_cpu(const int* a, const int* b, int dim) {
    // Fast-path common small dimensions
    if (dim == 3) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
    if (dim == 4) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
    }
    for (int i = 0; i < dim; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline int _search_hash_table_cpu(
    const int* table_kvs,        // [capacity * 2]
    const int* coords,           // [N * num_dims]
    const int* query_key,        // [num_dims]
    int table_capacity,
    int num_dims
) {
    int slot = CityHashCPU::hash(query_key, num_dims, table_capacity);
    const int begin = slot;
    int attempts = 0;
    while (attempts < table_capacity) {
        const int marker = table_kvs[slot * 2 + 0];
        if (marker == -1) {
            return -1; // empty => absent
        }
        const int vec_idx = table_kvs[slot * 2 + 1];
        if (vec_idx != -1) {
            const int* candidate = &coords[vec_idx * num_dims];
            if (_vec_equal_cpu(candidate, query_key, num_dims)) {
                return vec_idx;
            }
        }
        slot = (slot + 1) % table_capacity;
        if (slot == begin) return -1; // full cycle
        ++attempts;
    }
    return -1;
}

struct HashTableIntCPU {
    std::vector<int> table_kvs; // length = 2 * capacity
    int capacity = 0;
    const int* h_coords = nullptr; // external storage [num_coords * num_dims]
    int num_coords = 0;
    int num_dims = 3;

    inline void set_num_dims(int d) { num_dims = d; }
    inline int get_num_dims() const { return num_dims; }

    void resize(int new_capacity) {
        capacity = new_capacity;
        table_kvs.assign(static_cast<size_t>(capacity) * 2, -1);
    }

    void prepare() {
        if (capacity <= 0) return;
        std::fill(table_kvs.begin(), table_kvs.end(), -1);
    }

    void insert(const int* h_coords_in, int n_keys) {
        h_coords = h_coords_in;
        num_coords = n_keys;
        if (capacity <= 0 || n_keys <= 0) return;

        // For efficiency, keep local raw ptr
        int* kv = table_kvs.data();
        const int D = num_dims;
        for (int idx = 0; idx < n_keys; ++idx) {
            const int* key = &h_coords[idx * D];
            int slot = CityHashCPU::hash(key, D, capacity);
            const int begin = slot;
            int attempts = 0;
            while (attempts < capacity) {
                int& marker = kv[slot * 2 + 0];
                if (marker == -1) {
                    marker = slot;           // claim
                    kv[slot * 2 + 1] = idx;  // publish index
                    break;
                }
                slot = (slot + 1) % capacity;
                if (slot == begin) break; // table full
                ++attempts;
            }
        }
    }

    void build(const int* h_coords_in, int n_keys) {
        int desired_capacity = n_keys * 2;
        if (desired_capacity < 16) desired_capacity = 16;
        resize(desired_capacity);
        prepare();
        insert(h_coords_in, n_keys);
    }

    void search(const int* h_queries, int n_queries, int* h_out_indices) const {
        if (capacity <= 0 || n_queries <= 0) return;
        const int D = num_dims;
        const int* kv = table_kvs.data();
        for (int i = 0; i < n_queries; ++i) {
            const int* q = &h_queries[i * D];
            h_out_indices[i] = _search_hash_table_cpu(kv, h_coords, q, capacity, D);
        }
    }
};

} // namespace cubvh


