// Metal BVH query kernels — GPU-accelerated distance and ray queries.
// 4 kernels: unsigned_distance, signed_distance_watertight, signed_distance_raystab, raytrace.
#include <metal_stdlib>
using namespace metal;

#include "../bvh_types.h"

// ─── Utility functions ───────────────────────────────────────────────────────

inline float safe_divide(float num, float denom, float eps = BVH_EPSILON) {
    if (abs(denom) < eps) {
        return (denom <= 0) ? -(num / eps) : (num / eps);
    }
    return num / denom;
}

inline float sign_f(float x) {
    return copysign(1.0f, x);
}

inline float clamp_f(float val, float lo, float hi) {
    return min(max(val, lo), hi);
}

// ─── Triangle operations ─────────────────────────────────────────────────────

inline float3 tri_normal(float3 a, float3 b, float3 c) {
    return normalize(cross(b - a, c - a));
}

inline float tri_ray_intersect(float3 a, float3 b, float3 c, float3 ro, float3 rd) {
    // Möller-Trumbore
    float3 v1v0 = b - a;
    float3 v2v0 = c - a;
    float3 rov0 = ro - a;
    float3 n = cross(v1v0, v2v0);
    float3 q = cross(rov0, rd);
    float d = safe_divide(1.0f, dot(rd, n));
    float u = d * -dot(q, v2v0);
    float v = d * dot(q, v1v0);
    float t = d * -dot(n, rov0);
    if (u < 0.0f || u > 1.0f || v < 0.0f || (u + v) > 1.0f || t < 0.0f) t = 1e6f;
    return t;
}

inline float tri_distance_sq(float3 a, float3 b, float3 c, float3 pos) {
    float3 v21 = b - a; float3 p1 = pos - a;
    float3 v32 = c - b; float3 p2 = pos - b;
    float3 v13 = a - c; float3 p3 = pos - c;
    float3 nor = cross(v21, v13);

    float s1 = sign_f(dot(cross(v21, nor), p1));
    float s2 = sign_f(dot(cross(v32, nor), p2));
    float s3 = sign_f(dot(cross(v13, nor), p3));

    if (s1 + s2 + s3 < 2.0f) {
        // Closest to an edge
        float3 e1 = v21 * clamp_f(dot(v21, p1) / dot(v21, v21), 0.0f, 1.0f) - p1;
        float3 e2 = v32 * clamp_f(dot(v32, p2) / dot(v32, v32), 0.0f, 1.0f) - p2;
        float3 e3 = v13 * clamp_f(dot(v13, p3) / dot(v13, v13), 0.0f, 1.0f) - p3;
        return min(min(dot(e1, e1), dot(e2, e2)), dot(e3, e3));
    } else {
        // Closest to face
        float d = dot(nor, p1);
        return d * d / dot(nor, nor);
    }
}

inline bool tri_point_in_triangle(float3 a, float3 b, float3 c, float3 p) {
    float3 la = a - p, lb = b - p, lc = c - p;
    float3 u = cross(lb, lc);
    float3 v = cross(lc, la);
    float3 w = cross(la, lb);
    if (dot(u, v) < 0.0f) return false;
    if (dot(u, w) < 0.0f) return false;
    return true;
}

inline float3 tri_closest_point_to_line(float3 la, float3 lb, float3 lc) {
    float t = dot(lc - la, lb - la) / dot(lb - la, lb - la);
    t = clamp_f(t, 0.0f, 1.0f);
    return la + t * (lb - la);
}

inline float3 tri_closest_point(float3 a, float3 b, float3 c, float3 point) {
    // Check for degenerate triangle (zero-area) — fall back to closest-point-on-edges
    float3 edge1 = b - a;
    float3 edge2 = c - a;
    float3 n_raw = cross(edge1, edge2);
    if (length_squared(n_raw) < 1e-12f) {
        float3 c1 = tri_closest_point_to_line(a, b, point);
        float3 c2 = tri_closest_point_to_line(b, c, point);
        float3 c3 = tri_closest_point_to_line(c, a, point);
        float d1 = length_squared(point - c1);
        float d2 = length_squared(point - c2);
        float d3 = length_squared(point - c3);
        float mn = min(min(d1, d2), d3);
        if (mn == d1) return c1;
        if (mn == d2) return c2;
        return c3;
    }

    float3 n = normalize(n_raw);
    float3 proj = point - dot(n, point - a) * n;

    if (tri_point_in_triangle(a, b, c, proj)) return proj;

    float3 c1 = tri_closest_point_to_line(a, b, proj);
    float3 c2 = tri_closest_point_to_line(b, c, proj);
    float3 c3 = tri_closest_point_to_line(c, a, proj);

    float d1 = length_squared(proj - c1);
    float d2 = length_squared(proj - c2);
    float d3 = length_squared(proj - c3);

    float mn = min(min(d1, d2), d3);
    if (mn == d1) return c1;
    if (mn == d2) return c2;
    return c3;
}

inline float3 tri_barycentric(float3 a, float3 b, float3 c, float3 p) {
    float3 v0 = b - a;
    float3 v1 = c - a;
    float3 v2 = p - a;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    float denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-12f) return float3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return float3(u, v, w);
}

// ─── Bounding box operations ─────────────────────────────────────────────────

inline float bb_distance_sq(BvhBoundingBox bb, float3 p) {
    float3 d = max(bb.mn - p, max(p - bb.mx, float3(0.0f)));
    return dot(d, d);
}

inline float2 bb_ray_intersect(BvhBoundingBox bb, float3 pos, float3 dir) {
    float tmin = safe_divide(bb.mn.x - pos.x, dir.x);
    float tmax = safe_divide(bb.mx.x - pos.x, dir.x);
    if (tmin > tmax) { float tmp = tmin; tmin = tmax; tmax = tmp; }

    float tymin = safe_divide(bb.mn.y - pos.y, dir.y);
    float tymax = safe_divide(bb.mx.y - pos.y, dir.y);
    if (tymin > tymax) { float tmp = tymin; tymin = tymax; tymax = tmp; }

    if (tmin > tymax || tymin > tmax) return float2(FLT_MAX, FLT_MAX);
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    float tzmin = safe_divide(bb.mn.z - pos.z, dir.z);
    float tzmax = safe_divide(bb.mx.z - pos.z, dir.z);
    if (tzmin > tzmax) { float tmp = tzmin; tzmin = tzmax; tzmax = tmp; }

    if (tmin > tzmax || tzmin > tmax) return float2(FLT_MAX, FLT_MAX);
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    return float2(tmin, tmax);
}

// ─── Fixed-size stack ────────────────────────────────────────────────────────
// Depth 24 was NOT sufficient: this BVH is 4-ary and pushes up to 4 children per
// pop, so max stack occupancy ≈ 3·tree_depth+1. A multi-million-triangle mesh
// (e.g. the 8.6M-face decoded mesh) has BVH4 depth ~11-15, needing ~34-46 slots.
// At 24 the stack silently dropped pushes (`if count < 24`) → whole subtrees were
// skipped → closest_triangle missed the true nearest primitive and OVER-estimated
// distance (returned >0 even AT a vertex). That corrupted the narrow-band UDF and
// made remesh holey/fragmented. 64 covers BVH4 depth ~21 (4^21 leaves).

struct FixedStack64 {
    int elems[64];
    int count = 0;

    void push(int val) {
        if (count < 64) elems[count++] = val;
    }
    int pop() { return elems[--count]; }
    bool empty() { return count <= 0; }
};

// ─── Sorting network for child ordering ──────────────────────────────────────

struct DistAndIdx {
    float dist;
    uint idx;
};

inline void compare_and_swap(thread DistAndIdx& a, thread DistAndIdx& b) {
    if (a.dist < b.dist) {
        DistAndIdx tmp = a; a = b; b = tmp;
    }
}

inline void sort4(thread DistAndIdx v[4]) {
    compare_and_swap(v[0], v[2]);
    compare_and_swap(v[1], v[3]);
    compare_and_swap(v[0], v[1]);
    compare_and_swap(v[2], v[3]);
    compare_and_swap(v[1], v[2]);
}

// ─── BVH traversal ──────────────────────────────────────────────────────────

// Stackless closest triangle
inline int closest_triangle_stackless(
    float3 point,
    device const BvhNode* nodes,
    device const BvhTriangle* tris,
    thread float& out_dist_sq,
    float max_dist_sq
) {
    float shortest_dist_sq = max_dist_sq;
    int shortest_idx = -1;

    int idx = 0;
    while (idx != -1) {
        device const BvhNode& node = nodes[idx];
        float dbb = bb_distance_sq(node.bb, point);
        if (dbb > shortest_dist_sq) {
            idx = node.escape_idx;
            continue;
        }
        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float dsq = tri_distance_sq(tris[i].a, tris[i].b, tris[i].c, point);
                if (dsq <= shortest_dist_sq) {
                    shortest_dist_sq = dsq;
                    shortest_idx = i;
                }
            }
            idx = node.escape_idx;
        } else {
            idx = node.left_idx;
        }
    }

    if (shortest_idx == -1) {
        shortest_idx = 0;
        shortest_dist_sq = 0.0f;
    }
    out_dist_sq = shortest_dist_sq;
    return shortest_idx;
}

// Stack-based closest triangle (with sorting network, stack=24)
inline int closest_triangle(
    float3 point,
    device const BvhNode* nodes,
    device const BvhTriangle* tris,
    thread float& out_dist,
    float max_dist_sq = BVH_MAX_DIST_SQ
) {
    FixedStack64 stack;
    stack.push(0);

    float shortest_dist_sq = max_dist_sq;
    int shortest_idx = -1;

    while (!stack.empty()) {
        int idx = stack.pop();
        device const BvhNode& node = nodes[idx];

        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float dsq = tri_distance_sq(tris[i].a, tris[i].b, tris[i].c, point);
                if (dsq <= shortest_dist_sq) {
                    shortest_dist_sq = dsq;
                    shortest_idx = i;
                }
            }
        } else {
            DistAndIdx children[4];
            uint first_child = (uint)node.left_idx;
            for (uint i = 0; i < 4; ++i) {
                children[i] = {bb_distance_sq(nodes[i + first_child].bb, point), i + first_child};
            }
            sort4(children);
            for (uint i = 0; i < 4; ++i) {
                if (children[i].dist <= shortest_dist_sq) {
                    stack.push((int)children[i].idx);
                }
            }
        }
    }

    if (shortest_idx == -1) {
        shortest_idx = 0;
        shortest_dist_sq = 0.0f;
    }
    out_dist = sqrt(shortest_dist_sq);
    return shortest_idx;
}

// Stackless ray intersection
inline int ray_intersect_stackless(
    float3 ro, float3 rd,
    device const BvhNode* nodes,
    device const BvhTriangle* tris,
    thread float& out_t
) {
    float mint = BVH_MAX_DIST;
    int shortest_idx = -1;

    int idx = 0;
    while (idx != -1) {
        device const BvhNode& node = nodes[idx];
        float tbb = bb_ray_intersect(node.bb, ro, rd).x;
        if (tbb >= mint) {
            idx = node.escape_idx;
            continue;
        }
        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float t = tri_ray_intersect(tris[i].a, tris[i].b, tris[i].c, ro, rd);
                if (t < mint) { mint = t; shortest_idx = i; }
            }
            idx = node.escape_idx;
        } else {
            idx = node.left_idx;
        }
    }
    out_t = mint;
    return shortest_idx;
}

// Stack-based ray intersection (with sorting network, stack=24)
inline int ray_intersect(
    float3 ro, float3 rd,
    device const BvhNode* nodes,
    device const BvhTriangle* tris,
    thread float& out_t
) {
    FixedStack64 stack;
    stack.push(0);

    float mint = BVH_MAX_DIST;
    int shortest_idx = -1;

    while (!stack.empty()) {
        int idx = stack.pop();
        device const BvhNode& node = nodes[idx];

        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float t = tri_ray_intersect(tris[i].a, tris[i].b, tris[i].c, ro, rd);
                if (t < mint) { mint = t; shortest_idx = i; }
            }
        } else {
            DistAndIdx children[4];
            uint first_child = (uint)node.left_idx;
            for (uint i = 0; i < 4; ++i) {
                children[i] = {bb_ray_intersect(nodes[i + first_child].bb, ro, rd).x, i + first_child};
            }
            sort4(children);
            for (uint i = 0; i < 4; ++i) {
                if (children[i].dist < mint) {
                    stack.push((int)children[i].idx);
                }
            }
        }
    }

    out_t = mint;
    return shortest_idx;
}

// Average normal around a point (for watertight SDF)
inline float3 avg_normal_around_point(
    float3 point,
    device const BvhNode* nodes,
    device const BvhTriangle* tris
) {
    FixedStack64 stack;
    stack.push(0);

    float total_weight = 0.0f;
    float3 result = float3(0.0f);

    while (!stack.empty()) {
        int idx = stack.pop();
        device const BvhNode& node = nodes[idx];

        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                if (tri_distance_sq(tris[i].a, tris[i].b, tris[i].c, point) < BVH_EPSILON) {
                    result += tri_normal(tris[i].a, tris[i].b, tris[i].c);
                    total_weight += 1.0f;
                }
            }
        } else {
            uint first_child = (uint)node.left_idx;
            for (uint i = 0; i < 4; ++i) {
                if (bb_distance_sq(nodes[i + first_child].bb, point) < BVH_EPSILON) {
                    stack.push((int)(i + first_child));
                }
            }
        }
    }

    return result / max(total_weight, 1e-8f);
}

// ─── PCG32 RNG (for raystab) ────────────────────────────────────────────────

struct pcg32 {
    ulong state;
    ulong inc;

    pcg32() : state(0x853c49e6748fea9bUL), inc(0xda3e39cb94b95bdbUL) {}

    uint next_uint() {
        ulong oldstate = state;
        state = oldstate * 0x5851f42d4c957f2dUL + inc;
        uint xorshifted = (uint)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint rot = (uint)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    float next_float() {
        uint u = (next_uint() >> 9) | 0x3f800000u;
        return as_type<float>(u) - 1.0f;
    }

    void advance(long delta) {
        ulong cur_mult = 0x5851f42d4c957f2dUL;
        ulong cur_plus = inc;
        ulong acc_mult = 1u;
        ulong acc_plus = 0u;
        ulong d = (ulong)delta;
        while (d > 0) {
            if (d & 1) {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            d /= 2;
        }
        state = acc_mult * state + acc_plus;
    }
};

// Fibonacci lattice direction (for raystab)
inline float fractf(float x) { return x - floor(x); }

inline float3 cylindrical_to_dir(float2 p) {
    float cos_theta = -2.0f * p.x + 1.0f;
    float phi = 2.0f * M_PI_F * (p.y - 0.5f);
    float sin_theta = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

inline float3 fibonacci_dir_32(uint i, float2 offset) {
    // N_DIRS=32 → epsilon=1.33
    float epsilon = 1.33f;
    float golden_ratio = 1.6180339887498948f;
    return cylindrical_to_dir(float2(
        fractf((float(i) + epsilon) / (31.0f + 2.0f * epsilon) + offset.x),
        fractf(float(i) / golden_ratio + offset.y)
    ));
}

// ─── Global kernels ──────────────────────────────────────────────────────────
// NOTE: PyTorch tensors are packed [N,3] float32 = 12 bytes per row.
// Metal float3 is 16 bytes (padded). Use packed_float3 for tensor I/O.
//
// Dtype specialization: the 4 kernels below are macro-instantiated 3× each
// (float / half / bfloat) parametrizing only the scalar output buffer
// (distances or depth). Internal math always runs fp32 — only the final
// store is narrowed. Vec3 inputs/outputs (positions/uvw/rays) stay
// packed_float3 because Metal has no standard packed_half3 and per-component
// narrowing isn't worth the complexity for a first round.

#define UNSIGNED_DISTANCE_KERNEL(NAME, OUT_T)                        \
kernel void NAME(                                                     \
    device const packed_float3* positions [[buffer(0)]],             \
    device OUT_T* distances               [[buffer(1)]],             \
    device int64_t* face_id               [[buffer(2)]],             \
    device packed_float3* uvw             [[buffer(3)]],             \
    device const BvhNode* nodes           [[buffer(4)]],             \
    device const BvhTriangle* tris        [[buffer(5)]],             \
    constant uint& n_elements             [[buffer(6)]],             \
    constant uint& return_uvw             [[buffer(7)]],             \
    uint tid [[thread_position_in_grid]]                              \
) {                                                                   \
    if (tid >= n_elements) return;                                    \
                                                                      \
    float3 point = float3(positions[tid]);                            \
    float dist;                                                       \
    int idx = closest_triangle(point, nodes, tris, dist);             \
                                                                      \
    distances[tid] = (OUT_T)dist;                                     \
    face_id[tid]   = tris[idx].id;                                    \
                                                                      \
    if (return_uvw) {                                                 \
        float3 cp = tri_closest_point(tris[idx].a, tris[idx].b, tris[idx].c, point); \
        uvw[tid] = packed_float3(tri_barycentric(tris[idx].a, tris[idx].b, tris[idx].c, cp)); \
    }                                                                 \
}

UNSIGNED_DISTANCE_KERNEL(unsigned_distance_kernel,        float)
UNSIGNED_DISTANCE_KERNEL(unsigned_distance_kernel_half,   half)
UNSIGNED_DISTANCE_KERNEL(unsigned_distance_kernel_bfloat, bfloat)

#define SIGNED_DISTANCE_WATERTIGHT_KERNEL(NAME, OUT_T)               \
kernel void NAME(                                                     \
    device const packed_float3* positions [[buffer(0)]],             \
    device OUT_T* distances               [[buffer(1)]],             \
    device int64_t* face_id               [[buffer(2)]],             \
    device packed_float3* uvw             [[buffer(3)]],             \
    device const BvhNode* nodes           [[buffer(4)]],             \
    device const BvhTriangle* tris        [[buffer(5)]],             \
    constant uint& n_elements             [[buffer(6)]],             \
    constant uint& return_uvw             [[buffer(7)]],             \
    uint tid [[thread_position_in_grid]]                              \
) {                                                                   \
    if (tid >= n_elements) return;                                    \
                                                                      \
    float3 point = float3(positions[tid]);                            \
    float dist;                                                       \
    int idx = closest_triangle(point, nodes, tris, dist);             \
                                                                      \
    float3 cp = tri_closest_point(tris[idx].a, tris[idx].b, tris[idx].c, point); \
    float3 avg_n = avg_normal_around_point(cp, nodes, tris);          \
    float sign_val = dot(avg_n, point - cp) >= 0.0f ? 1.0f : -1.0f;   \
                                                                      \
    distances[tid] = (OUT_T)(sign_val * dist);                        \
    face_id[tid]   = tris[idx].id;                                    \
                                                                      \
    if (return_uvw) {                                                 \
        uvw[tid] = packed_float3(tri_barycentric(tris[idx].a, tris[idx].b, tris[idx].c, cp)); \
    }                                                                 \
}

SIGNED_DISTANCE_WATERTIGHT_KERNEL(signed_distance_watertight_kernel,        float)
SIGNED_DISTANCE_WATERTIGHT_KERNEL(signed_distance_watertight_kernel_half,   half)
SIGNED_DISTANCE_WATERTIGHT_KERNEL(signed_distance_watertight_kernel_bfloat, bfloat)

#define SIGNED_DISTANCE_RAYSTAB_KERNEL(NAME, OUT_T)                  \
kernel void NAME(                                                     \
    device const packed_float3* positions [[buffer(0)]],             \
    device OUT_T* distances               [[buffer(1)]],             \
    device int64_t* face_id               [[buffer(2)]],             \
    device packed_float3* uvw             [[buffer(3)]],             \
    device const BvhNode* nodes           [[buffer(4)]],             \
    device const BvhTriangle* tris        [[buffer(5)]],             \
    constant uint& n_elements             [[buffer(6)]],             \
    constant uint& return_uvw             [[buffer(7)]],             \
    uint tid [[thread_position_in_grid]]                              \
) {                                                                   \
    if (tid >= n_elements) return;                                    \
                                                                      \
    float3 point = float3(positions[tid]);                            \
    float dist;                                                       \
    int idx = closest_triangle(point, nodes, tris, dist);             \
                                                                      \
    pcg32 rng;                                                        \
    rng.advance((long)tid * 2);                                       \
    float2 offset = float2(rng.next_float(), rng.next_float());       \
                                                                      \
    bool inside = true;                                               \
    for (uint i = 0; i < BVH_N_STAB_RAYS; ++i) {                      \
        float3 d = fibonacci_dir_32(i, offset);                       \
        float t1, t2;                                                 \
        int hit1 = ray_intersect(point, -d, nodes, tris, t1);         \
        int hit2 = ray_intersect(point,  d, nodes, tris, t2);         \
        if (hit1 < 0 || hit2 < 0) { inside = false; break; }          \
    }                                                                 \
                                                                      \
    distances[tid] = (OUT_T)(inside ? -dist : dist);                  \
    face_id[tid]   = tris[idx].id;                                    \
                                                                      \
    if (return_uvw) {                                                 \
        float3 cp = tri_closest_point(tris[idx].a, tris[idx].b, tris[idx].c, point); \
        uvw[tid] = packed_float3(tri_barycentric(tris[idx].a, tris[idx].b, tris[idx].c, cp)); \
    }                                                                 \
}

SIGNED_DISTANCE_RAYSTAB_KERNEL(signed_distance_raystab_kernel,        float)
SIGNED_DISTANCE_RAYSTAB_KERNEL(signed_distance_raystab_kernel_half,   half)
SIGNED_DISTANCE_RAYSTAB_KERNEL(signed_distance_raystab_kernel_bfloat, bfloat)

#define RAYTRACE_KERNEL(NAME, OUT_T)                                 \
kernel void NAME(                                                     \
    device const packed_float3* rays_o    [[buffer(0)]],             \
    device const packed_float3* rays_d    [[buffer(1)]],             \
    device packed_float3* positions       [[buffer(2)]],             \
    device int64_t* face_id               [[buffer(3)]],             \
    device OUT_T* depth                   [[buffer(4)]],             \
    device const BvhNode* nodes           [[buffer(5)]],             \
    device const BvhTriangle* tris        [[buffer(6)]],             \
    constant uint& n_elements             [[buffer(7)]],             \
    uint tid [[thread_position_in_grid]]                              \
) {                                                                   \
    if (tid >= n_elements) return;                                    \
                                                                      \
    float3 ro = float3(rays_o[tid]);                                  \
    float3 rd = float3(rays_d[tid]);                                  \
                                                                      \
    float t;                                                          \
    int idx = ray_intersect(ro, rd, nodes, tris, t);                  \
                                                                      \
    depth[tid]     = (OUT_T)t;                                        \
    positions[tid] = packed_float3(ro + t * rd);                      \
    face_id[tid]   = (idx >= 0) ? tris[idx].id : (int64_t)-1;         \
}

RAYTRACE_KERNEL(raytrace_kernel,        float)
RAYTRACE_KERNEL(raytrace_kernel_half,   half)
RAYTRACE_KERNEL(raytrace_kernel_bfloat, bfloat)
