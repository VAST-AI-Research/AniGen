#ifndef HILBERT_H
#define HILBERT_H

#include <metal_stdlib>
using namespace metal;
#include "utils.h"

// Hilbert encode for uint32 codes
inline void hilbert_encode_32(uint b, uint x, uint y, uint z, uint bit_length, thread uint& code) {
    uint point[3] = {x, y, z};
    uint m = 1u << (bit_length - 1);
    uint q, p, t;

    // Inverse undo excess work
    q = m;
    while (q > 1) {
        p = q - 1;
        for (int i = 0; i < 3; i++) {
            if (point[i] & q) {
                point[0] ^= p;
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

    uint xx = expandBits32(point[0]);
    uint yy = expandBits32(point[1]);
    uint zz = expandBits32(point[2]);
    uint c_code = xx * 4 + yy * 2 + zz;
    uint mask = (0xFFFFFFFFu >> (32 - 3 * bit_length));
    code = (c_code & mask) | (b << (3 * bit_length));
}

// Hilbert encode for uint64 codes
inline void hilbert_encode_64(uint b, uint x, uint y, uint z, uint bit_length, thread ulong& code) {
    uint point[3] = {x, y, z};
    uint m = 1u << (bit_length - 1);
    uint q, p, t;

    q = m;
    while (q > 1) {
        p = q - 1;
        for (int i = 0; i < 3; i++) {
            if (point[i] & q) {
                point[0] ^= p;
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q >>= 1;
    }

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

    ulong xx = expandBits64((ulong)point[0]);
    ulong yy = expandBits64((ulong)point[1]);
    ulong zz = expandBits64((ulong)point[2]);
    ulong c_code = xx * 4 + yy * 2 + zz;
    ulong mask = (0xFFFFFFFFFFFFFFFFull >> (64 - 3 * bit_length));
    code = (c_code & mask) | ((ulong)b << (3 * bit_length));
}

// Hilbert decode for uint32 codes
inline void hilbert_decode_32(uint code, uint bit_length, thread uint& b, thread uint& x, thread uint& y, thread uint& z) {
    uint mask = (0xFFFFFFFFu >> (32 - 3 * bit_length));
    uint c_code = (code & mask);
    b = code >> (3 * bit_length);

    uint point[3];
    point[0] = extractBits32(c_code >> 2);
    point[1] = extractBits32(c_code >> 1);
    point[2] = extractBits32(c_code);

    uint m = 2u << (bit_length - 1);
    uint q, p, t;

    // Gray decode
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

// Hilbert decode for uint64 codes
inline void hilbert_decode_64(ulong code, uint bit_length, thread uint& b, thread uint& x, thread uint& y, thread uint& z) {
    ulong mask = (0xFFFFFFFFFFFFFFFFull >> (64 - 3 * bit_length));
    ulong c_code = (code & mask);
    b = (uint)(code >> (3 * bit_length));

    uint point[3];
    point[0] = (uint)extractBits64(c_code >> 2);
    point[1] = (uint)extractBits64(c_code >> 1);
    point[2] = (uint)extractBits64(c_code);

    uint m = 2u << (bit_length - 1);
    uint q, p, t;

    t = point[2] >> 1;
    for (int i = 2; i > 0; i--) {
        point[i] ^= point[i - 1];
    }
    point[0] ^= t;

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

#endif // HILBERT_H
