/*
 * Hashmap
 *
 * Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <torch/extension.h>


#define BLOCK_SIZE 256


namespace cumesh {


/**
 * Get the active vetices of a sparse voxel grid
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as voxel indices
 * @param coords        [M, 3] int32 tensor containing the coordinates of the active voxels
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 *  
 * @return              [L, 3] int32 tensor containing the active vertices
 */
torch::Tensor get_sparse_voxel_grid_active_vertices(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const int W,
    const int H,
    const int D
);


/**
 * Isosurfacing a volume defined on vertices of a sparse voxel grid using a simple dual contouring algorithm.
 * Dual vertices are computed by mean of edge intersections.
 * 
 * @param hashmap_keys  [Nvert] uint32/uint64 hashmap of the vertices keys
 * @param hashmap_vals  [Nvert] uint32 tensor containing the hashmap values as vertex indices
 * @param coords        [Mvox, 3] int32 tensor containing the coordinates of the active voxels
 * @param udf           [Mvert] float tensor containing the UDF/SDF values at the vertices
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 *
 * @return              [L, 3] float tensor containing the active vertices (Dual Vertices)
                        [L, 3] int32 tensor containing the intersected edges (1: intersected, 0: not intersected)
 */
std::tuple<torch::Tensor, torch::Tensor> simple_dual_contour(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& udf,
    int W,
    int H,
    int D
);


} // namespace cumesh