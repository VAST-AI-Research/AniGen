#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Weighted sum forward: output[m, c] = sum_v input[indices[m,v], c] * weight[m,v]
// ============================================================================

kernel void indice_weighted_sum_fwd(
    const device float* input     [[buffer(0)]],   // [N, C]
    const device uint* indices    [[buffer(1)]],   // [M, V]
    const device float* weights   [[buffer(2)]],   // [M, V]
    device float* output          [[buffer(3)]],   // [M, C]
    constant uint& M              [[buffer(4)]],
    constant uint& C              [[buffer(5)]],
    constant uint& V              [[buffer(6)]],
    uint2 tid                     [[thread_position_in_grid]]
) {
    uint m = tid.y;
    uint c = tid.x;
    if (m >= M || c >= C) return;

    float acc = 0.0f;
    for (uint v = 0; v < V; v++) {
        uint idx = indices[m * V + v];
        if (idx != 0xFFFFFFFFu) {
            acc += input[idx * C + c] * weights[m * V + v];
        }
    }
    output[m * C + c] = acc;
}

// ============================================================================
// Weighted sum backward input:
//   grad_input[indices[m,v], c] += grad_output[m, c] * weight[m, v]
// Uses atomic_fetch_add for scatter-add
// ============================================================================

kernel void indice_weighted_sum_bwd_input(
    const device float* grad_output   [[buffer(0)]],   // [M, C]
    const device uint* indices        [[buffer(1)]],   // [M, V]
    const device float* weights       [[buffer(2)]],   // [M, V]
    device atomic_float* grad_input   [[buffer(3)]],   // [N, C]
    constant uint& M                  [[buffer(4)]],
    constant uint& C                  [[buffer(5)]],
    constant uint& V                  [[buffer(6)]],
    uint2 tid                         [[thread_position_in_grid]]
) {
    // Grid: (C, M * V) — one thread per (c, m, v) triple
    uint c = tid.x;
    uint mv = tid.y;
    uint m = mv / V;
    uint v = mv % V;
    if (m >= M || c >= C) return;

    uint idx = indices[m * V + v];
    if (idx == 0xFFFFFFFFu) return;

    float contrib = grad_output[m * C + c] * weights[m * V + v];
    atomic_fetch_add_explicit(&grad_input[idx * C + c], contrib, memory_order_relaxed);
}
