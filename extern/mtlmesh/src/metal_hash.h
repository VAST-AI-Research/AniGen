// Host-side dispatch for Metal spatial hash table kernels
#pragma once

#import "metal_context.h"
#import <torch/torch.h>

namespace mtlmesh {

// Insert 3D coords as keys, thread indices as values.
// keys: [M] uint32 or uint64 hashmap keys (pre-allocated, filled with max)
// vals: [M] uint32 hashmap values (pre-allocated)
// coords: [N, 4] int32 (padding, x, y, z) — column 0 ignored
// W, H, D: grid dimensions
void hashmap_insert_3d_idx_as_val(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, int W, int H, int D);

// Lookup 3D coords in hashmap.
// keys/vals: hashmap state from insert
// coords: [N, 4] int32 query coordinates
// Returns: [N] uint32 values (0xffffffff if not found)
torch::Tensor hashmap_lookup_3d(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, int W, int H, int D);

// Get active vertices for sparse voxel grid.
// keys/vals: hashmap with voxels inserted
// coords: [Nvox, 3] int32 voxel coordinates
// Returns: [Nvert, 3] int32 unique vertex coordinates
torch::Tensor get_sparse_voxel_grid_active_vertices(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, int W, int H, int D);

// Simple dual contouring on sparse voxel grid.
// keys/vals: hashmap with vertices inserted
// coords: [Nvox, 3] int32 voxel coordinates
// distances: [Nvert] float unsigned distances
// Returns: tuple(vertices [Nvert, 3] float, intersected [Nvox, 3] int)
std::tuple<torch::Tensor, torch::Tensor> simple_dual_contour(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, torch::Tensor distances,
    int W, int H, int D);

} // namespace mtlmesh
