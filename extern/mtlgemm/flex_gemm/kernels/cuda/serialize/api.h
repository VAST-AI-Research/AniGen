/*
 * Serialize a voxel grid
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
namespace serialize {

/**
 * Z-order encode 3D points
 *
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 * @param bit_length    bit length of the coordinates
 * @param codes         [N] tensor containing the z-order encoded values
 */
void z_order_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
);


/**
 * Z-order decode 3D points
 *
 * @param codes         [N] tensor containing the z-order encoded values
 * @param bit_length    bit length of the coordinates
 *
 * @return [N, 4] tensor containing the b, x, y, z coordinates
 */
torch::Tensor z_order_decode(
    const torch::Tensor& codes,
    const size_t bit_length
);


/**
 * Hilbert encode 3D points
 *
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 * @param bit_length    bit length of the coordinates
 * @param codes         [N] tensor containing the Hilbert encoded values
 */
void hilbert_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
);


/**
 * Hilbert decode 3D points
 *
 * @param codes         [N] tensor containing the Hilbert encoded values
 * @param bit_length    bit length of the coordinates
 *
 * @return [N, 4] tensor containing the b, x, y, z coordinates
 */
torch::Tensor hilbert_decode(
    const torch::Tensor& codes,
    const size_t bit_length
);

} // namespace serialize
} // namespace flex_gemm
