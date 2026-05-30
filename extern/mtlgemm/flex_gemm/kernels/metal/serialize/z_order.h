#ifndef Z_ORDER_H
#define Z_ORDER_H

#include <metal_stdlib>
using namespace metal;
#include "utils.h"

// Z-order encode for uint32 codes
inline void z_order_encode_32(uint b, uint x, uint y, uint z, uint bit_length, thread uint& code) {
    uint xx = expandBits32(x);
    uint yy = expandBits32(y);
    uint zz = expandBits32(z);
    uint c_code = xx * 4 + yy * 2 + zz;
    uint mask = (0xFFFFFFFFu >> (32 - 3 * bit_length));
    code = (c_code & mask) | (b << (3 * bit_length));
}

// Z-order encode for uint64 codes
inline void z_order_encode_64(uint b, uint x, uint y, uint z, uint bit_length, thread ulong& code) {
    ulong xx = expandBits64((ulong)x);
    ulong yy = expandBits64((ulong)y);
    ulong zz = expandBits64((ulong)z);
    ulong c_code = xx * 4 + yy * 2 + zz;
    ulong mask = (0xFFFFFFFFFFFFFFFFull >> (64 - 3 * bit_length));
    code = (c_code & mask) | ((ulong)b << (3 * bit_length));
}

// Z-order decode for uint32 codes
inline void z_order_decode_32(uint code, uint bit_length, thread uint& b, thread uint& x, thread uint& y, thread uint& z) {
    uint mask = (0xFFFFFFFFu >> (32 - 3 * bit_length));
    uint c_code = (code & mask);
    x = extractBits32(c_code >> 2);
    y = extractBits32(c_code >> 1);
    z = extractBits32(c_code);
    b = code >> (3 * bit_length);
}

// Z-order decode for uint64 codes
inline void z_order_decode_64(ulong code, uint bit_length, thread uint& b, thread uint& x, thread uint& y, thread uint& z) {
    ulong mask = (0xFFFFFFFFFFFFFFFFull >> (64 - 3 * bit_length));
    ulong c_code = (code & mask);
    x = (uint)extractBits64(c_code >> 2);
    y = (uint)extractBits64(c_code >> 1);
    z = (uint)extractBits64(c_code);
    b = (uint)(code >> (3 * bit_length));
}

#endif // Z_ORDER_H
