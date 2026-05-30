#ifndef SERIALIZE_UTILS_H
#define SERIALIZE_UTILS_H

#include <metal_stdlib>
using namespace metal;

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
inline uint expandBits32(uint v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

// Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
inline ulong expandBits64(ulong v) {
    v = (v | (v << 32)) & 0x001F00000000FFFFull;
    v = (v | (v << 16)) & 0x001F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}

// Removes 2 zeros after each bit in a 30-bit integer.
inline uint extractBits32(uint v) {
    v = v & 0x09249249u;
    v = (v ^ (v >>  2)) & 0x030C30C3u;
    v = (v ^ (v >>  4)) & 0x0300F00Fu;
    v = (v ^ (v >>  8)) & 0x030000FFu;
    v = (v ^ (v >> 16)) & 0x000003FFu;
    return v;
}

// Removes 2 zeros after each bit in a 63-bit integer.
inline ulong extractBits64(ulong v) {
    v = v & 0x1249249249249249ull;
    v = (v ^ (v >>  2)) & 0x10C30C30C30C30C3ull;
    v = (v ^ (v >>  4)) & 0x100F00F00F00F00Full;
    v = (v ^ (v >>  8)) & 0x001F0000FF0000FFull;
    v = (v ^ (v >> 16)) & 0x001F00000000FFFFull;
    v = (v ^ (v >> 32)) & 0x00000000001FFFFFull;
    return v;
}

#endif // SERIALIZE_UTILS_H
