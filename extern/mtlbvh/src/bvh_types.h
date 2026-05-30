// Shared BVH types — used by both CPU (C++) and GPU (Metal) code.
// Plain C structs with float3 from simd — shared by CPU and Metal.
#pragma once

#ifdef __METAL_VERSION__
// Metal shader side — float3 is built-in
#include <metal_stdlib>
using namespace metal;
#else
// CPU/Obj-C++ side — use simd float3
#include <simd/simd.h>
typedef simd_float3 float3;
typedef simd_float2 float2;
#endif

// Triangle: 3 vertices + original face ID.
struct BvhTriangle {
    float3 a, b, c;
    int64_t id;
};

// Axis-aligned bounding box.
struct BvhBoundingBox {
    float3 mn;  // min (avoid keyword conflict in Metal)
    float3 mx;  // max
};

// BVH tree node.
// Leaves: left_idx < 0, encoding -(start_idx)-1; right_idx encodes -(end_idx)-1.
// Internal: left_idx = first child index; right_idx = one-past-last child.
// escape_idx: next node for stackless traversal (-1 = done).
struct BvhNode {
    BvhBoundingBox bb;
    int left_idx;
    int right_idx;
    int escape_idx;
};

// Constants
#ifdef __METAL_VERSION__
constant float BVH_MAX_DIST = 1000.0f;
constant float BVH_MAX_DIST_SQ = 1000000.0f;
constant float BVH_EPSILON = 1e-6f;
constant uint BVH_N_STAB_RAYS = 32;
#else
static constexpr float BVH_MAX_DIST = 1000.0f;
static constexpr float BVH_MAX_DIST_SQ = 1000000.0f;
static constexpr float BVH_EPSILON = 1e-6f;
static constexpr uint32_t BVH_N_STAB_RAYS = 32;
#endif
