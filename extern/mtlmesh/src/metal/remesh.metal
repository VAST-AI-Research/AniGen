// Remesh compute kernels — sparse voxel grid active vertex extraction and dual contouring
#include <metal_stdlib>
using namespace metal;


// ──────────────────────────────────────────────────────────────────────
// Inline hash functions
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

inline uint linear_probing_lookup_u32(
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

inline uint linear_probing_lookup_u64(
    const device ulong* hashmap_keys,
    const device uint* hashmap_values,
    ulong key, uint N
) {
    uint slot = hash64(key, N);
    for (uint probes = 0; probes < N; ++probes) {
        ulong prev = hashmap_keys[slot];
        if (prev == 0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFF;
        if (prev == key) return hashmap_values[slot];
        slot = (slot + 1 < N) ? slot + 1 : 0;
    }
    return 0xFFFFFFFF;  // table full / not found — never spin the GPU
}


// ──────────────────────────────────────────────────────────────────────
// get_vertex_num — count unique vertices from voxel corners
// For each active voxel, count 1 (itself) + number of 8 corner neighbors
// that are NOT in the hashmap (i.e. missing neighbors = new vertices)
// ──────────────────────────────────────────────────────────────────────

kernel void get_vertex_num_u32_kernel(
    const device uint*    hashmap_keys   [[buffer(0)]],
    const device uint*    hashmap_vals   [[buffer(1)]],
    const device int*     coords         [[buffer(2)]],
    device int*           num_vertices   [[buffer(3)]],
    constant uint&        N              [[buffer(4)]],
    constant uint&        M              [[buffer(5)]],
    constant int&         W              [[buffer(6)]],
    constant int&         H              [[buffer(7)]],
    constant int&         D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    int num = 1;  // include the current voxel

    int x = coords[3 * tid + 0];
    int y = coords[3 * tid + 1];
    int z = coords[3 * tid + 2];

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            for (int k = 0; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                int xx = x + i;
                int yy = y + j;
                int zz = z + k;
                if (xx >= W || yy >= H || zz >= D) {
                    num++;
                    continue;
                }
                uint flat_idx = uint(ulong(xx * H + yy) * ulong(D) + ulong(zz));
                if (linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, N) == 0xFFFFFFFF) {
                    num++;
                }
            }
        }
    }

    num_vertices[tid] = num;
}

kernel void get_vertex_num_u64_kernel(
    const device ulong*   hashmap_keys   [[buffer(0)]],
    const device uint*    hashmap_vals   [[buffer(1)]],
    const device int*     coords         [[buffer(2)]],
    device int*           num_vertices   [[buffer(3)]],
    constant uint&        N              [[buffer(4)]],
    constant uint&        M              [[buffer(5)]],
    constant int&         W              [[buffer(6)]],
    constant int&         H              [[buffer(7)]],
    constant int&         D              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    int num = 1;

    int x = coords[3 * tid + 0];
    int y = coords[3 * tid + 1];
    int z = coords[3 * tid + 2];

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            for (int k = 0; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                int xx = x + i;
                int yy = y + j;
                int zz = z + k;
                if (xx >= W || yy >= H || zz >= D) {
                    num++;
                    continue;
                }
                ulong flat_idx = ulong(xx * H + yy) * ulong(D) + ulong(zz);
                if (linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, N) == 0xFFFFFFFF) {
                    num++;
                }
            }
        }
    }

    num_vertices[tid] = num;
}


// ──────────────────────────────────────────────────────────────────────
// set_vertex — write unique vertex positions
// ──────────────────────────────────────────────────────────────────────

kernel void set_vertex_u32_kernel(
    const device uint*    hashmap_keys    [[buffer(0)]],
    const device uint*    hashmap_vals    [[buffer(1)]],
    const device int*     coords          [[buffer(2)]],
    const device int*     vertices_offset [[buffer(3)]],
    device int*           vertices        [[buffer(4)]],
    constant uint&        N               [[buffer(5)]],
    constant uint&        M               [[buffer(6)]],
    constant int&         W               [[buffer(7)]],
    constant int&         H               [[buffer(8)]],
    constant int&         D               [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    int x = coords[3 * tid + 0];
    int y = coords[3 * tid + 1];
    int z = coords[3 * tid + 2];
    int ptr_start = vertices_offset[tid];
    vertices[3 * ptr_start + 0] = x;
    vertices[3 * ptr_start + 1] = y;
    vertices[3 * ptr_start + 2] = z;
    ptr_start++;

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            for (int k = 0; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                int xx = x + i;
                int yy = y + j;
                int zz = z + k;
                if (xx >= W || yy >= H || zz >= D) {
                    vertices[3 * ptr_start + 0] = xx;
                    vertices[3 * ptr_start + 1] = yy;
                    vertices[3 * ptr_start + 2] = zz;
                    ptr_start++;
                    continue;
                }
                uint flat_idx = uint(ulong(xx * H + yy) * ulong(D) + ulong(zz));
                if (linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, N) == 0xFFFFFFFF) {
                    vertices[3 * ptr_start + 0] = xx;
                    vertices[3 * ptr_start + 1] = yy;
                    vertices[3 * ptr_start + 2] = zz;
                    ptr_start++;
                }
            }
        }
    }
}

kernel void set_vertex_u64_kernel(
    const device ulong*   hashmap_keys    [[buffer(0)]],
    const device uint*    hashmap_vals    [[buffer(1)]],
    const device int*     coords          [[buffer(2)]],
    const device int*     vertices_offset [[buffer(3)]],
    device int*           vertices        [[buffer(4)]],
    constant uint&        N               [[buffer(5)]],
    constant uint&        M               [[buffer(6)]],
    constant int&         W               [[buffer(7)]],
    constant int&         H               [[buffer(8)]],
    constant int&         D               [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    int x = coords[3 * tid + 0];
    int y = coords[3 * tid + 1];
    int z = coords[3 * tid + 2];
    int ptr_start = vertices_offset[tid];
    vertices[3 * ptr_start + 0] = x;
    vertices[3 * ptr_start + 1] = y;
    vertices[3 * ptr_start + 2] = z;
    ptr_start++;

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            for (int k = 0; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                int xx = x + i;
                int yy = y + j;
                int zz = z + k;
                if (xx >= W || yy >= H || zz >= D) {
                    vertices[3 * ptr_start + 0] = xx;
                    vertices[3 * ptr_start + 1] = yy;
                    vertices[3 * ptr_start + 2] = zz;
                    ptr_start++;
                    continue;
                }
                ulong flat_idx = ulong(xx * H + yy) * ulong(D) + ulong(zz);
                if (linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, N) == 0xFFFFFFFF) {
                    vertices[3 * ptr_start + 0] = xx;
                    vertices[3 * ptr_start + 1] = yy;
                    vertices[3 * ptr_start + 2] = zz;
                    ptr_start++;
                }
            }
        }
    }
}


// ──────────────────────────────────────────────────────────────────────
// Dual contouring helper — look up SDF value at a vertex
// ──────────────────────────────────────────────────────────────────────

inline float get_vertex_val_u32(
    const device uint* hashmap_keys,
    const device uint* hashmap_vals,
    const device float* udf,
    uint N_vert,
    uint M,
    int x, int y, int z,
    int W, int H, int D
) {
    uint flat_idx = uint(ulong(x) * ulong(H) * ulong(D) + ulong(y) * ulong(D) + ulong(z));
    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);
    return udf[idx];
}

inline float get_vertex_val_u64(
    const device ulong* hashmap_keys,
    const device uint* hashmap_vals,
    const device float* udf,
    uint N_vert,
    uint M,
    int x, int y, int z,
    int W, int H, int D
) {
    ulong flat_idx = ulong(x) * ulong(H) * ulong(D) + ulong(y) * ulong(D) + ulong(z);
    uint idx = linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, M);
    return udf[idx];
}


// ──────────────────────────────────────────────────────────────────────
// simple_dual_contour_kernel — compute dual contour vertex positions
// and intersection flags for each voxel
// ──────────────────────────────────────────────────────────────────────

kernel void simple_dual_contour_u32_kernel(
    const device uint*    hashmap_keys    [[buffer(0)]],
    const device uint*    hashmap_vals    [[buffer(1)]],
    const device int*     coords          [[buffer(2)]],
    const device float*   udf            [[buffer(3)]],
    device float*         out_vertices    [[buffer(4)]],
    device int*           out_intersected [[buffer(5)]],
    constant uint&        N_vert          [[buffer(6)]],
    constant uint&        M               [[buffer(7)]],
    constant int&         W               [[buffer(8)]],
    constant int&         H               [[buffer(9)]],
    constant int&         D               [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    int vx = coords[tid * 3 + 0];
    int vy = coords[tid * 3 + 1];
    int vz = coords[tid * 3 + 2];

    float3 intersection_sum = float3(0.0f, 0.0f, 0.0f);
    int intersection_count = 0;

    // Axis X
    for (int u = 0; u <= 1; u++) {
        for (int v = 0; v <= 1; v++) {
            float val1 = get_vertex_val_u32(hashmap_keys, hashmap_vals, udf, N_vert, M, vx, vy + u, vz + v, W, H, D);
            float val2 = get_vertex_val_u32(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + 1, vy + u, vz + v, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += float(vx) + t;
                intersection_sum.y += float(vy + u);
                intersection_sum.z += float(vz + v);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[tid * 3 + 0] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[tid * 3 + 0] = -1;
                } else {
                    out_intersected[tid * 3 + 0] = 0;
                }
            }
        }
    }

    // Axis Y
    for (int u = 0; u <= 1; u++) {
        for (int v = 0; v <= 1; v++) {
            float val1 = get_vertex_val_u32(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy, vz + v, W, H, D);
            float val2 = get_vertex_val_u32(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy + 1, vz + v, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += float(vx + u);
                intersection_sum.y += float(vy) + t;
                intersection_sum.z += float(vz + v);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[tid * 3 + 1] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[tid * 3 + 1] = -1;
                } else {
                    out_intersected[tid * 3 + 1] = 0;
                }
            }
        }
    }

    // Axis Z
    for (int u = 0; u <= 1; u++) {
        for (int v = 0; v <= 1; v++) {
            float val1 = get_vertex_val_u32(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy + v, vz, W, H, D);
            float val2 = get_vertex_val_u32(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy + v, vz + 1, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += float(vx + u);
                intersection_sum.y += float(vy + v);
                intersection_sum.z += float(vz) + t;
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[tid * 3 + 2] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[tid * 3 + 2] = -1;
                } else {
                    out_intersected[tid * 3 + 2] = 0;
                }
            }
        }
    }

    // Mean intersection point
    if (intersection_count > 0) {
        out_vertices[tid * 3 + 0] = intersection_sum.x / float(intersection_count);
        out_vertices[tid * 3 + 1] = intersection_sum.y / float(intersection_count);
        out_vertices[tid * 3 + 2] = intersection_sum.z / float(intersection_count);
    } else {
        // Fallback: voxel center
        out_vertices[tid * 3 + 0] = float(vx) + 0.5f;
        out_vertices[tid * 3 + 1] = float(vy) + 0.5f;
        out_vertices[tid * 3 + 2] = float(vz) + 0.5f;
    }
}

kernel void simple_dual_contour_u64_kernel(
    const device ulong*   hashmap_keys    [[buffer(0)]],
    const device uint*    hashmap_vals    [[buffer(1)]],
    const device int*     coords          [[buffer(2)]],
    const device float*   udf            [[buffer(3)]],
    device float*         out_vertices    [[buffer(4)]],
    device int*           out_intersected [[buffer(5)]],
    constant uint&        N_vert          [[buffer(6)]],
    constant uint&        M               [[buffer(7)]],
    constant int&         W               [[buffer(8)]],
    constant int&         H               [[buffer(9)]],
    constant int&         D               [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= M) return;

    int vx = coords[tid * 3 + 0];
    int vy = coords[tid * 3 + 1];
    int vz = coords[tid * 3 + 2];

    float3 intersection_sum = float3(0.0f, 0.0f, 0.0f);
    int intersection_count = 0;

    // Axis X
    for (int u = 0; u <= 1; u++) {
        for (int v = 0; v <= 1; v++) {
            float val1 = get_vertex_val_u64(hashmap_keys, hashmap_vals, udf, N_vert, M, vx, vy + u, vz + v, W, H, D);
            float val2 = get_vertex_val_u64(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + 1, vy + u, vz + v, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += float(vx) + t;
                intersection_sum.y += float(vy + u);
                intersection_sum.z += float(vz + v);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[tid * 3 + 0] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[tid * 3 + 0] = -1;
                } else {
                    out_intersected[tid * 3 + 0] = 0;
                }
            }
        }
    }

    // Axis Y
    for (int u = 0; u <= 1; u++) {
        for (int v = 0; v <= 1; v++) {
            float val1 = get_vertex_val_u64(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy, vz + v, W, H, D);
            float val2 = get_vertex_val_u64(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy + 1, vz + v, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += float(vx + u);
                intersection_sum.y += float(vy) + t;
                intersection_sum.z += float(vz + v);
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[tid * 3 + 1] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[tid * 3 + 1] = -1;
                } else {
                    out_intersected[tid * 3 + 1] = 0;
                }
            }
        }
    }

    // Axis Z
    for (int u = 0; u <= 1; u++) {
        for (int v = 0; v <= 1; v++) {
            float val1 = get_vertex_val_u64(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy + v, vz, W, H, D);
            float val2 = get_vertex_val_u64(hashmap_keys, hashmap_vals, udf, N_vert, M, vx + u, vy + v, vz + 1, W, H, D);

            if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
                float t = -val1 / (val2 - val1);
                intersection_sum.x += float(vx + u);
                intersection_sum.y += float(vy + v);
                intersection_sum.z += float(vz) + t;
                intersection_count++;
            }

            if (u == 1 && v == 1) {
                if (val1 < 0 && val2 >= 0) {
                    out_intersected[tid * 3 + 2] = 1;
                } else if (val1 >= 0 && val2 < 0) {
                    out_intersected[tid * 3 + 2] = -1;
                } else {
                    out_intersected[tid * 3 + 2] = 0;
                }
            }
        }
    }

    // Mean intersection point
    if (intersection_count > 0) {
        out_vertices[tid * 3 + 0] = intersection_sum.x / float(intersection_count);
        out_vertices[tid * 3 + 1] = intersection_sum.y / float(intersection_count);
        out_vertices[tid * 3 + 2] = intersection_sum.z / float(intersection_count);
    } else {
        out_vertices[tid * 3 + 0] = float(vx) + 0.5f;
        out_vertices[tid * 3 + 1] = float(vy) + 0.5f;
        out_vertices[tid * 3 + 2] = float(vz) + 0.5f;
    }
}
