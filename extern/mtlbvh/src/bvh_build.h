// CPU BVH tree construction — plain C++, no external dependencies.
// Median-split with branching factor 4, escape link threading.
#pragma once

#include "bvh_types.h"
#include <vector>
#include <cstdint>

// Build a BVH from triangles. Reorders triangles in place.
// Returns the node array. After building, triangles are sorted by the BVH leaf order.
std::vector<BvhNode> bvh_build(std::vector<BvhTriangle>& triangles, uint32_t n_primitives_per_leaf = 8);
