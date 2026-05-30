/*
 * Neighbor map for sparse submanifold convolution
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


namespace flex_gemm {
namespace spconv {

/**
 * Build sparse submanifold convolution neighbor map with hashmap
 * 
 * @param hashmap_keys  [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_vals  [N] uint32 tensor containing the hashmap values as tensor indices
 * @param coords        [M, 4] int32 tensor containing the keys to be looked up
 * @param W             the number of width dimensions
 * @param H             the number of height dimensions
 * @param D             the number of depth dimensions
 * @param Kw            the number of width kernel dimensions
 * @param Kh            the number of height kernel dimensions
 * @param Kd            the number of depth kernel dimensions
 * @param Dw            the dialation of width
 * @param Dh            the dialation of height
 * @param Dd            the dialation of depth
 *  
 * @return              [M, Kw * Kh * Kd] uint32 tensor containing the submanifold convolution neighbor map
 */
torch::Tensor hashmap_build_submanifold_conv_neighbour_map_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    int W,
    int H,
    int D,
    int Kw,
    int Kh,
    int Kd,
    int Dw,
    int Dh,
    int Dd
);


/**
 * Convert neighbor map to gray and binary code
 * 
 * @param neighbor_map     [N, V] uint32 tensor containing the neighbor map
 * 
 * @return                 [N] gray code
 *                         [N] sorted idx
 *                         [N] valid signal idx for input tensor
 *                         [N] valid signal idx for output tensor
 *                         [V+1] valid signal segment
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_1(
    const torch::Tensor& neighbor_map
);


/**
 * Get valid kernel indices for masked implicit gemm
 * 
 * @param gray_code     [N] gray code
 * @param sorted_idx    [N] sorted idx
 * @param block_size    the block size of CUDA kernel (must be power of 2)
 * 
 * @return              [L] uint32 tensor containing the valid kernel indices
 *                      [num_blocks + 1] uint32 tensor containing the segment of valid kernel indices
 */
std::tuple<torch::Tensor, torch::Tensor> neighbor_map_post_process_for_masked_implicit_gemm_2(
    const torch::Tensor& gray_code,
    const torch::Tensor& sorted_idx,
    int block_size
);

} // namespace spconv
} // namespace flex_gemm
