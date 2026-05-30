#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


namespace flex_gemm {
namespace serialize {

/**
 * Hilbert encode 3D points
 *
 * @param b     batch index
 * @param x     x coordinate
 * @param y     y coordinate
 * @param z     z coordinate
 * @param bit_length  bit length of the coordinates
 * @param code  hilbert encoded value
 */
template<typename T>
__host__ __device__ __forceinline__ void hilbert_encode(uint32_t b, uint32_t x, uint32_t y, uint32_t z, size_t bit_length, T& code) {
    uint32_t point[3] = {x, y, z};
    uint32_t m = 1 << (bit_length - 1);
    uint32_t q, p, t;

    // Inverse undo excess work
    q = m;
    while (q > 1) {
        p = q - 1;
        for (int i = 0; i < 3; i++) {
            if (point[i] & q) {
                point[0] ^= p;  // invert
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q >>= 1;
    }

    // Gray encode
    for (int i = 1; i < 3; i++) {
        point[i] ^= point[i - 1];
    }
    t = 0;
    q = m;
    while (q > 1) {
        if (point[2] & q) {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for (int i = 0; i < 3; i++) {
        point[i] ^= t;
    }

    // Convert transformed coordinates to Morton code (which represents Hilbert index here)
    T xx = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[0]));
    T yy = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[1]));
    T zz = flex_gemm::serialize::utils::expandBits(static_cast<T>(point[2]));
    
    T c_code = xx * 4 + yy * 2 + zz;
    T mask = (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    
    // Combine with batch index
    code = (c_code & mask) | (static_cast<T>(b) << (3 * bit_length));
}


/**
 * Hilbert decode 3D points
 *
 * @param code  hilbert encoded value
 * @param bit_length  bit length of the coordinates
 * @param b     decoded batch index
 * @param x     decoded x coordinate
 * @param y     decoded y coordinate
 * @param z     decoded z coordinate
 */
template<typename T>
__host__ __device__ __forceinline__ void hilbert_decode(T code, size_t bit_length, uint32_t& b, uint32_t& x, uint32_t& y, uint32_t& z) {
    // Extract Batch and Spatial Code
    T mask = (std::numeric_limits<T>::max() >> (sizeof(T) * 8 - 3 * bit_length));
    T c_code = (code & mask);
    b = static_cast<uint32_t>(code >> (3 * bit_length));

    uint32_t point[3];
    // Morton decode to get back the transformed coordinates
    point[0] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code >> 2));
    point[1] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code >> 1));
    point[2] = static_cast<uint32_t>(flex_gemm::serialize::utils::extractBits(c_code));

    uint32_t m = 2 << (bit_length - 1); // equivalent to 1 << bit_length but handling loop bound
    uint32_t q, p, t;

    // Gray decode by H ^ (H/2)
    t = point[2] >> 1;
    for (int i = 2; i > 0; i--) {
        point[i] ^= point[i - 1];
    }
    point[0] ^= t;

    // Undo excess work
    q = 2;
    while (q != m) {
        p = q - 1;
        for (int i = 2; i >= 0; i--) {
            if (point[i] & q) {
                point[0] ^= p;
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q <<= 1;
    }

    x = point[0];
    y = point[1];
    z = point[2];
}

namespace cuda {

/**
 * Hilbert encode 3D points
 *
 * @param N             length of the input tensors
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 * @param bit_length    bit length of the coordinates
 * @param codes         [N] tensor containing the hilbert encoded values
 */
template<typename T>
__global__ void hilbert_encode(
    const size_t N,
    const uint32_t* coords,
    const size_t bit_length,
    T* codes
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= N) return;
    
    // Vectorized load for performance
    uint4 coord = *(reinterpret_cast<const uint4*>(coords + 4 * thread_id));
    T code;
    flex_gemm::serialize::hilbert_encode(coord.x, coord.y, coord.z, coord.w, bit_length, code);
    codes[thread_id] = code;
}


/**
 * Hilbert decode 3D points
 *
 * @param N             length of the input tensors
 * @param codes         [N] tensor containing the hilbert encoded values
 * @param bit_length    bit length of the coordinates
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 */
template<typename T>
__global__ void hilbert_decode(
    const size_t N,
    const T* codes,
    const size_t bit_length,
    uint32_t* coords
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= N) return;

    uint32_t b, x, y, z;
    flex_gemm::serialize::hilbert_decode(codes[thread_id], bit_length, b, x, y, z);
    // Vectorized store
    *(reinterpret_cast<uint4*>(coords + 4 * thread_id)) = make_uint4(b, x, y, z);
}

} // namespace cuda

namespace cpu {

/**
 * Hilbert encode 3D points
 *
 * @param N             length of the input tensors
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 * @param bit_length    bit length of the coordinates
 * @param codes         [N] tensor containing the hilbert encoded values
 */
template<typename T>
__host__ void hilbert_encode(
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
        flex_gemm::serialize::hilbert_encode(b, x, y, z, bit_length, code);
        codes[thread_id] = code;
    }
}


/**
 * Hilbert decode 3D points
 *
 * @param N             length of the input tensors
 * @param codes         [N] tensor containing the hilbert encoded values
 * @param bit_length    bit length of the coordinates
 * @param coords        [N, 4] tensor containing the b, x, y, z coordinates
 */
template<typename T>
__host__ void hilbert_decode(
    const size_t N,
    const T* codes,
    const size_t bit_length,
    uint32_t* coords
) {
    #pragma omp parallel for schedule(static)
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        T code = codes[thread_id];
        uint32_t b, x, y, z;
        flex_gemm::serialize::hilbert_decode(code, bit_length, b, x, y, z);
        coords[4 * thread_id + 0] = b;
        coords[4 * thread_id + 1] = x;
        coords[4 * thread_id + 2] = y;
        coords[4 * thread_id + 3] = z;
    }
}

} // namespace cpu
} // namespace serialize
} // namespace flex_gemm
