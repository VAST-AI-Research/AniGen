// CPU BVH tree construction — plain C++, no external dependencies.
// Median-split with branching factor 4 and escape link threading.
#include "bvh_build.h"
#include <stack>
#include <array>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <functional>

static constexpr int BRANCHING_FACTOR = 4;

// Helper: component-wise min/max for float3
static inline float3 f3min(float3 a, float3 b) {
    return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}
static inline float3 f3max(float3 a, float3 b) {
    return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

static inline float3 tri_centroid(const BvhTriangle& t) {
    return {(t.a.x + t.b.x + t.c.x) / 3.0f,
            (t.a.y + t.b.y + t.c.y) / 3.0f,
            (t.a.z + t.b.z + t.c.z) / 3.0f};
}

static inline float f3_component(float3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

// Compute bounding box from a range of triangles.
static BvhBoundingBox compute_bb(std::vector<BvhTriangle>::iterator begin,
                                  std::vector<BvhTriangle>::iterator end) {
    float3 mn = begin->a;
    float3 mx = begin->a;
    for (auto it = begin; it != end; ++it) {
        mn = f3min(mn, it->a); mx = f3max(mx, it->a);
        mn = f3min(mn, it->b); mx = f3max(mx, it->b);
        mn = f3min(mn, it->c); mx = f3max(mx, it->c);
    }
    return {mn, mx};
}

std::vector<BvhNode> bvh_build(std::vector<BvhTriangle>& triangles, uint32_t n_primitives_per_leaf) {
    std::vector<BvhNode> nodes;

    // Root
    nodes.emplace_back();
    nodes.front().bb = compute_bb(triangles.begin(), triangles.end());

    struct BuildNode {
        int node_idx;
        std::vector<BvhTriangle>::iterator begin;
        std::vector<BvhTriangle>::iterator end;
    };

    std::stack<BuildNode> build_stack;
    build_stack.push({0, triangles.begin(), triangles.end()});

    while (!build_stack.empty()) {
        const BuildNode curr = build_stack.top();
        int node_idx = curr.node_idx;

        std::array<BuildNode, BRANCHING_FACTOR> children;
        children[0].begin = curr.begin;
        children[0].end = curr.end;

        build_stack.pop();

        // Partition triangles into BRANCHING_FACTOR children
        int n_children = 1;
        while (n_children < BRANCHING_FACTOR) {
            for (int i = n_children - 1; i >= 0; --i) {
                auto& child = children[i];
                size_t count = std::distance(child.begin, child.end);

                // Choose axis with maximum variance
                float3 mean = {0, 0, 0};
                for (auto it = child.begin; it != child.end; ++it) {
                    float3 c = tri_centroid(*it);
                    mean.x += c.x; mean.y += c.y; mean.z += c.z;
                }
                float inv_n = 1.0f / (float)count;
                mean.x *= inv_n; mean.y *= inv_n; mean.z *= inv_n;

                float3 var = {0, 0, 0};
                for (auto it = child.begin; it != child.end; ++it) {
                    float3 c = tri_centroid(*it);
                    float dx = c.x - mean.x, dy = c.y - mean.y, dz = c.z - mean.z;
                    var.x += dx * dx; var.y += dy * dy; var.z += dz * dz;
                }

                // Pick axis with max variance
                int axis = 0;
                float max_var = var.x;
                if (var.y > max_var) { max_var = var.y; axis = 1; }
                if (var.z > max_var) { axis = 2; }

                auto m = child.begin + count / 2;
                std::nth_element(child.begin, m, child.end,
                    [axis](const BvhTriangle& t1, const BvhTriangle& t2) {
                        return f3_component(tri_centroid(t1), axis) < f3_component(tri_centroid(t2), axis);
                    });

                children[i * 2].begin = children[i].begin;
                children[i * 2 + 1].end = children[i].end;
                children[i * 2].end = children[i * 2 + 1].begin = m;
            }
            n_children *= 2;
        }

        // Create child nodes
        nodes[node_idx].left_idx = (int)nodes.size();
        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
            auto& child = children[i];
            assert(child.begin != child.end);
            child.node_idx = (int)nodes.size();

            BvhNode new_node;
            new_node.bb = compute_bb(child.begin, child.end);
            new_node.left_idx = 0;
            new_node.right_idx = 0;
            new_node.escape_idx = -1;

            if ((size_t)std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
                new_node.left_idx = -(int)std::distance(triangles.begin(), child.begin) - 1;
                new_node.right_idx = -(int)std::distance(triangles.begin(), child.end) - 1;
            } else {
                // Will be filled when this node is popped from build_stack
            }

            nodes.push_back(new_node);

            if ((size_t)std::distance(child.begin, child.end) > n_primitives_per_leaf) {
                build_stack.push(child);
            }
        }
        nodes[node_idx].right_idx = (int)nodes.size();
    }

    // Thread BVH with escape links for stackless traversal
    for (auto& n : nodes) n.escape_idx = -1;

    std::function<void(int, int)> thread_bvh = [&](int node_idx, int escape_idx) {
        BvhNode& node = nodes[node_idx];
        node.escape_idx = escape_idx;
        if (node.left_idx < 0) return;  // leaf
        int first_child = node.left_idx;
        int end_child = node.right_idx;  // exclusive
        for (int c = first_child; c < end_child; ++c) {
            int next_escape = (c + 1 < end_child) ? (c + 1) : escape_idx;
            thread_bvh(c, next_escape);
        }
    };

    if (!nodes.empty()) {
        thread_bvh(0, -1);
    }

    return nodes;
}
