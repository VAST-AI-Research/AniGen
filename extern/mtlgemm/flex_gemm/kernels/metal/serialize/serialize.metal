#include <metal_stdlib>
using namespace metal;

#include "z_order.h"
#include "hilbert.h"

// ============================================================================
// Z-order encode/decode kernels
// ============================================================================

kernel void z_order_encode_3d_u32(
    const device uint* coords   [[buffer(0)]],  // [N, 4] as flat uint (b, x, y, z)
    device uint* codes          [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b = coords[tid * 4 + 0];
    uint x = coords[tid * 4 + 1];
    uint y = coords[tid * 4 + 2];
    uint z = coords[tid * 4 + 3];
    uint code;
    z_order_encode_32(b, x, y, z, bit_length, code);
    codes[tid] = code;
}

kernel void z_order_encode_3d_u64(
    const device uint* coords   [[buffer(0)]],
    device ulong* codes         [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b = coords[tid * 4 + 0];
    uint x = coords[tid * 4 + 1];
    uint y = coords[tid * 4 + 2];
    uint z = coords[tid * 4 + 3];
    ulong code;
    z_order_encode_64(b, x, y, z, bit_length, code);
    codes[tid] = code;
}

kernel void z_order_decode_3d_u32(
    const device uint* codes    [[buffer(0)]],
    device uint* coords         [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b, x, y, z;
    z_order_decode_32(codes[tid], bit_length, b, x, y, z);
    coords[tid * 4 + 0] = b;
    coords[tid * 4 + 1] = x;
    coords[tid * 4 + 2] = y;
    coords[tid * 4 + 3] = z;
}

kernel void z_order_decode_3d_u64(
    const device ulong* codes   [[buffer(0)]],
    device uint* coords         [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b, x, y, z;
    z_order_decode_64(codes[tid], bit_length, b, x, y, z);
    coords[tid * 4 + 0] = b;
    coords[tid * 4 + 1] = x;
    coords[tid * 4 + 2] = y;
    coords[tid * 4 + 3] = z;
}

// ============================================================================
// Hilbert encode/decode kernels
// ============================================================================

kernel void hilbert_encode_3d_u32(
    const device uint* coords   [[buffer(0)]],
    device uint* codes          [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b = coords[tid * 4 + 0];
    uint x = coords[tid * 4 + 1];
    uint y = coords[tid * 4 + 2];
    uint z = coords[tid * 4 + 3];
    uint code;
    hilbert_encode_32(b, x, y, z, bit_length, code);
    codes[tid] = code;
}

kernel void hilbert_encode_3d_u64(
    const device uint* coords   [[buffer(0)]],
    device ulong* codes         [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b = coords[tid * 4 + 0];
    uint x = coords[tid * 4 + 1];
    uint y = coords[tid * 4 + 2];
    uint z = coords[tid * 4 + 3];
    ulong code;
    hilbert_encode_64(b, x, y, z, bit_length, code);
    codes[tid] = code;
}

kernel void hilbert_decode_3d_u32(
    const device uint* codes    [[buffer(0)]],
    device uint* coords         [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b, x, y, z;
    hilbert_decode_32(codes[tid], bit_length, b, x, y, z);
    coords[tid * 4 + 0] = b;
    coords[tid * 4 + 1] = x;
    coords[tid * 4 + 2] = y;
    coords[tid * 4 + 3] = z;
}

kernel void hilbert_decode_3d_u64(
    const device ulong* codes   [[buffer(0)]],
    device uint* coords         [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& bit_length   [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    uint b, x, y, z;
    hilbert_decode_64(codes[tid], bit_length, b, x, y, z);
    coords[tid * 4 + 0] = b;
    coords[tid * 4 + 1] = x;
    coords[tid * 4 + 2] = y;
    coords[tid * 4 + 3] = z;
}
