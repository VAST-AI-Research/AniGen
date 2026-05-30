#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


namespace flex_gemm {
namespace serialize {

/**
 * Z-order encode 3D points
 *
 * @param b     batch index
 * @param x     x coordinate
 * @param y     y coordinate
 * @param z     z coordinate
 * @param bit_length  bit length of the coordinates
 * @param code  z-order encoded value
 */
template<typename T>
__host__ __device__ __forceinline__ void z_order_encode(uint32_t b, uint32_t x, uint32_t y, uint32_t z, size_t bit_length, T& code) {
    T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(x));
    T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(y));
    T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(z));
    T c_code = xx * 4 + yy * 2 + zz;
    T mask = (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    code = (c_code & mask) | (b << (3 * bit_length));
}


/**
 * Z-order decode 3D points
 *
 * @param code  z-order encoded value
 * @param bit_length  bit length of the coordinates
 * @param b     decoded batch index
 * @param x     decoded x coordinate
 * @param y     decoded y coordinate
 * @param z     decoded z coordinate
 */
template<typename T>
__host__ __device__ __forceinline__ void z_order_decode(T code, size_t bit_length, uint32_t& b, uint32_t& x, uint32_t& y, uint32_t& z) {
    T mask = (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    T c_code = (code & mask);
    x = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code >> 2));
    y = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code >> 1));
    z = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code));
    b = static_cast<uint32_t>(code >> (3 * bit_length));
}

namespace cuda {

/**
 * Z-order encode 3D points
 *
 * @param N             length of the input tensors
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 * @param bit_length    bit length of the coordinates
 * @param codes         [N] tensor containing the z-order encoded values
 */
template<typename T>
__global__ void z_order_encode(
    const size_t N,
    const uint32_t* coords,
    const size_t bit_length,
    T* codes
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id >= N) return;
    uint4 coord = *(reinterpret_cast<const uint4*>(coords + 4 * thread_id));
    T code;
    flex_gemm::serialize::z_order_encode(coord.x, coord.y, coord.z, coord.w, bit_length, code);
    codes[thread_id] = code;
}


/**
 * Z-order decode 3D points
 *
 * @param N             length of the input tensors
 * @param codes         [N] tensor containing the z-order encoded values
 * @param bit_length    bit length of the coordinates
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 */
template<typename T>
__global__ void z_order_decode(
    const size_t N,
    const T* codes,
    const size_t bit_length,
    uint32_t* coords
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= N) return;

    uint32_t b, x, y, z;
    flex_gemm::serialize::z_order_decode(codes[thread_id], bit_length, b, x, y, z);
    *(reinterpret_cast<uint4*>(coords + 4 * thread_id)) = make_uint4(b, x, y, z);
}

} // namespace cuda

namespace cpu {
    
/**
 * Z-order encode 3D points
 *
 * @param N             length of the input tensors
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 * @param bit_length    bit length of the coordinates
 * @param codes         [N] tensor containing the z-order encoded values
 */
template<typename T>
__host__ void z_order_encode(
    const size_t N,
    const uint32_t* coords,
    const size_t bit_length,
    T* codes
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        uint32_t b = coords[4 * thread_id + 0];
        uint32_t x = coords[4 * thread_id + 1];
        uint32_t y = coords[4 * thread_id + 2];
        uint32_t z = coords[4 * thread_id + 3];
        T code;
        flex_gemm::serialize::z_order_encode(b, x, y, z, bit_length, code);
        codes[thread_id] = code;
    }
}


/**
 * Z-order decode 3D points
 *
 * @param N             length of the input tensors
 * @param codes         [N] tensor containing the z-order encoded values
 * @param bit_length    bit length of the coordinates
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 */
template<typename T>
__host__ void z_order_decode(
    const size_t N,
    const T* codes,
    const size_t bit_length,
    uint32_t* coords
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        T code = codes[thread_id];
        uint32_t b, x, y, z;
        flex_gemm::serialize::z_order_decode(code, bit_length, b, x, y, z);
        coords[4 * thread_id + 0] = b;
        coords[4 * thread_id + 1] = x;
        coords[4 * thread_id + 2] = y;
        coords[4 * thread_id + 3] = z;
    }
}

} // namespace cpu
} // namespace serialize
} // namespace flex_gemm
