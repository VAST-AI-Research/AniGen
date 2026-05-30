#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


namespace flex_gemm {
namespace serialize {
namespace utils {

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__host__ __device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}


// Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
__host__ __device__ __forceinline__ uint64_t expandBits(uint64_t v) {
    v = (v | (v << 32)) & 0x001F00000000FFFFull;
    v = (v | (v << 16)) & 0x001F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}


// Removes 2 zeros after each bit in a 30-bit integer.
__host__ __device__ __forceinline__ uint32_t extractBits(uint32_t v) {
    v = v & 0x09249249;
    v = (v ^ (v >>  2)) & 0x030C30C3u;
    v = (v ^ (v >>  4)) & 0x0300F00Fu;
    v = (v ^ (v >>  8)) & 0x030000FFu;
    v = (v ^ (v >> 16)) & 0x000003FFu;
    return v;
}


// Removes 2 zeros after each bit in a 63-bit integer.
__host__ __device__ __forceinline__ uint64_t extractBits(uint64_t v) {
    v = v & 0x1249249249249249ull;
    v = (v ^ (v >>  2)) & 0x10C30C30C30C30C3ull;
    v = (v ^ (v >>  4)) & 0x100F00F00F00F00Full;
    v = (v ^ (v >>  8)) & 0x001F0000FF0000FFull;
    v = (v ^ (v >> 16)) & 0x001F00000000FFFFull;
    v = (v ^ (v >> 32)) & 0x00000000001FFFFFull;
    return v;
}

} // namespace utils
} // namespace serialize
} // namespace flex_gemm
