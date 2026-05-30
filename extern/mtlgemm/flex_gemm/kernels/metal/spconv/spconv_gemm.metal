#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "config.h"

// ============================================================================
// Implicit GEMM for sparse submanifold convolution
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8x8 matmul.
// Three kernels (each with float32, half, bfloat16 specializations):
//   forward, backward-input, backward-weight.
//
// Block sizes: B1=64 (N-tile), B2=64 (channel-tile), BK=32 (K-tile)
// 256 threads = 8 SIMD groups, each owns one row of 8x8 sub-tiles
// Threadgroup memory holds tiles in the *input* dtype (so half/bfloat halve it).
// Accumulators are always float32 — only the loads/stores narrow.
// ============================================================================

#define B1 GEMM_BLOCK_N   // 64
#define B2 GEMM_BLOCK_CO  // 64
#define BK GEMM_BLOCK_K   // 32
#define SENTINEL 0xFFFFFFFFu

// Number of 8x8 sub-tiles per dimension
#define TILES_M (B1 / 8)   // 8
#define TILES_N (B2 / 8)   // 8

// ============================================================================
// Forward: output[N, Co] = sum_v input[neighbor[n,v], :] * weight[co, v, :]
//
// Grid: (cdiv(N, B1), cdiv(Co, B2))
// Threadgroup: 256 threads (8 SIMD groups)
// ============================================================================

kernel void spconv_fwd_implicit_gemm(
    const device float* input         [[buffer(0)]],   // [N, Ci]
    const device float* weight        [[buffer(1)]],   // [Co, V, Ci]
    const device float* bias          [[buffer(2)]],   // [Co] or empty
    const device uint*  neighbor      [[buffer(3)]],   // [N, V]
    device float*       output        [[buffer(4)]],   // [N, Co]
    constant uint& N                  [[buffer(5)]],
    constant uint& Co                 [[buffer(6)]],
    constant uint& Ci                 [[buffer(7)]],
    constant uint& V                  [[buffer(8)]],
    constant uint& has_bias           [[buffer(9)]],
    threadgroup float* smem           [[threadgroup(0)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint  lid                         [[thread_index_in_threadgroup]],
    uint  simd_id                     [[simdgroup_index_in_threadgroup]],
    uint  lane_id                     [[thread_index_in_simdgroup]]
) {
    uint n_base  = gid.x * B1;
    uint co_base = gid.y * B2;

    uint threads = GEMM_THREADS;

    // Shared memory layout: smem_a[B1][BK] | smem_b[BK][B2] | smem_nb[B1] | smem_any_valid (1 atomic_uint)
    threadgroup float* smem_a  = smem;
    threadgroup float* smem_b  = smem + B1 * BK;
    threadgroup uint*  smem_nb = (threadgroup uint*)(smem + B1 * BK + BK * B2);
    threadgroup atomic_uint* smem_any_valid = (threadgroup atomic_uint*)(smem_nb + B1);

    // Each SIMD group owns one row of 8x8 sub-tiles (TILES_N accumulators)
    // simd_id selects which row (0..TILES_M-1)
    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Loop over volume elements
    for (uint v = 0; v < V; v++) {
        if (lid == 0) {
            atomic_store_explicit(smem_any_valid, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load B1 neighbor indices into shared memory + skip-empty-V vote
        for (uint i = lid; i < B1; i += threads) {
            uint n = n_base + i;
            uint nb = (n < N) ? neighbor[n * V + v] : SENTINEL;
            smem_nb[i] = nb;
            if (nb != SENTINEL) {
                atomic_fetch_or_explicit(smem_any_valid, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(smem_any_valid, memory_order_relaxed) == 0) {
            continue;  // No row in this block has a neighbor at v — skip the entire Ci loop.
        }

        // Loop over Ci in BK-sized blocks
        for (uint bk = 0; bk < Ci; bk += BK) {
            // Cooperative load: smem_a[B1][BK] = input[neighbor[n, v], bk..bk+BK]
            for (uint i = lid; i < B1 * BK; i += threads) {
                uint row = i / BK;
                uint col = i % BK;
                uint ci_idx = bk + col;
                uint nb_idx = smem_nb[row];
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?
                    input[nb_idx * Ci + ci_idx] : 0.0f;
            }

            // Cooperative load: smem_b[BK][B2] = weight[co, v, bk..bk+BK]
            // weight layout: [Co, V, Ci] — weight[co, v, ci] = weight[(co * V + v) * Ci + ci]
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;
                uint col = i % B2;
                uint ci_idx = bk + row;
                uint co_idx = co_base + col;
                smem_b[row * B2 + col] = (ci_idx < Ci && co_idx < Co) ?
                    weight[(co_idx * V + v) * Ci + ci_idx] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // simdgroup_matrix multiply: each SIMD group does its row of 8x8 tiles
            simdgroup_matrix<float, 8, 8> a_mat;
            for (uint k8 = 0; k8 < BK; k8 += 8) {
                // Load A sub-tile: rows [simd_id*8..simd_id*8+7], cols [k8..k8+7]
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
                for (uint tn = 0; tn < TILES_N; tn++) {
                    simdgroup_matrix<float, 8, 8> b_mat;
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results: write accumulators to shared memory, then global with bounds + bias
    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write from shared memory to global with bounds checking + bias
    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint n = n_base + row;
        uint co = co_base + col;
        if (n < N && co < Co) {
            float val = smem_a[row * B2 + col];
            if (has_bias) val += bias[co];
            output[n * Co + co] = val;
        }
    }
}

// ============================================================================
// Backward-input: grad_input[N, Ci] = sum_v grad_output[neighbor_inv[n,v], :] * weight^T
//
// neighbor_inv[n, v] = neighbor[n, V-1-v]  (flipped)
//
// Grid: (cdiv(N, B1), cdiv(Ci, B2))
// Threadgroup: 256 threads (8 SIMD groups)
// ============================================================================

kernel void spconv_bwd_input_implicit_gemm(
    const device float* grad_output   [[buffer(0)]],   // [N, Co]
    const device float* weight        [[buffer(1)]],   // [Co, V, Ci]
    const device uint*  neighbor      [[buffer(2)]],   // [N, V]
    device float*       grad_input    [[buffer(3)]],   // [N, Ci]
    constant uint& N                  [[buffer(4)]],
    constant uint& Co                 [[buffer(5)]],
    constant uint& Ci                 [[buffer(6)]],
    constant uint& V                  [[buffer(7)]],
    threadgroup float* smem           [[threadgroup(0)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint  lid                         [[thread_index_in_threadgroup]],
    uint  simd_id                     [[simdgroup_index_in_threadgroup]],
    uint  lane_id                     [[thread_index_in_simdgroup]]
) {
    uint n_base  = gid.x * B1;
    uint ci_base = gid.y * B2;

    uint threads = GEMM_THREADS;

    threadgroup float* smem_a  = smem;
    threadgroup float* smem_b  = smem + B1 * BK;
    threadgroup uint*  smem_nb = (threadgroup uint*)(smem + B1 * BK + BK * B2);

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint v = 0; v < V; v++) {
        // Flipped neighbor access: neighbor[n, V-1-v]
        uint v_flip = V - 1 - v;

        for (uint i = lid; i < B1; i += threads) {
            uint n = n_base + i;
            smem_nb[i] = (n < N) ? neighbor[n * V + v_flip] : SENTINEL;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bk = 0; bk < Co; bk += BK) {
            // smem_a[B1][BK] = grad_output[neighbor_inv[n, v], bk..bk+BK]
            for (uint i = lid; i < B1 * BK; i += threads) {
                uint row = i / BK;
                uint col = i % BK;
                uint co_idx = bk + col;
                uint nb_idx = smem_nb[row];
                smem_a[row * BK + col] = (nb_idx != SENTINEL && co_idx < Co) ?
                    grad_output[nb_idx * Co + co_idx] : 0.0f;
            }

            // smem_b[BK][B2] = weight^T for volume v
            // We want smem_b[k][ci] = weight[bk+k, v, ci_base+ci]
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;
                uint col = i % B2;
                uint co_idx = bk + row;
                uint ci_idx = ci_base + col;
                smem_b[row * B2 + col] = (co_idx < Co && ci_idx < Ci) ?
                    weight[(co_idx * V + v) * Ci + ci_idx] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // simdgroup_matrix multiply
            simdgroup_matrix<float, 8, 8> a_mat;
            for (uint k8 = 0; k8 < BK; k8 += 8) {
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
                for (uint tn = 0; tn < TILES_N; tn++) {
                    simdgroup_matrix<float, 8, 8> b_mat;
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results to shared memory then global
    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint n = n_base + row;
        uint ci = ci_base + col;
        if (n < N && ci < Ci) {
            grad_input[n * Ci + ci] = smem_a[row * B2 + col];
        }
    }
}

// ============================================================================
// Backward-weight: grad_weight[Co, V, Ci] = sum_n grad_output[n, co] * input[neighbor[n, v], ci]
//
// Grid: (cdiv(Co, B1), cdiv(V * Ci, B2))
// K dimension = N (iterate over all voxels)
// Threadgroup: 256 threads (8 SIMD groups)
// ============================================================================

kernel void spconv_bwd_weight_implicit_gemm(
    const device float* grad_output   [[buffer(0)]],   // [N, Co]
    const device float* input         [[buffer(1)]],   // [N, Ci]
    const device uint*  neighbor      [[buffer(2)]],   // [N, V]
    device float*       grad_weight   [[buffer(3)]],   // [Co, V, Ci]
    constant uint& N                  [[buffer(4)]],
    constant uint& Co                 [[buffer(5)]],
    constant uint& Ci                 [[buffer(6)]],
    constant uint& V                  [[buffer(7)]],
    threadgroup float* smem           [[threadgroup(0)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint  lid                         [[thread_index_in_threadgroup]],
    uint  simd_id                     [[simdgroup_index_in_threadgroup]],
    uint  lane_id                     [[thread_index_in_simdgroup]]
) {
    uint co_base = gid.x * B1;
    uint vci_base = gid.y * B2;  // flat index into V*Ci

    uint threads = GEMM_THREADS;
    uint VCi = V * Ci;

    // Shared memory: smem_a[B1][BK] for grad_output, smem_b[BK][B2] for input via neighbor
    threadgroup float* smem_a = smem;
    threadgroup float* smem_b = smem + B1 * BK;

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    // K dimension = N, iterate in BK-sized blocks
    for (uint bn = 0; bn < N; bn += BK) {
        // smem_a[B1][BK] = grad_output^T[co, n] = grad_output[bn+k, co_base+row]
        for (uint i = lid; i < B1 * BK; i += threads) {
            uint row = i / BK;  // co offset
            uint col = i % BK;  // n offset
            uint co_idx = co_base + row;
            uint n_idx = bn + col;
            smem_a[row * BK + col] = (co_idx < Co && n_idx < N) ?
                grad_output[n_idx * Co + co_idx] : 0.0f;
        }

        // smem_b[BK][B2] = input gathered via neighbor
        // vci_base + col maps to (v, ci): v = idx / Ci, ci = idx % Ci
        for (uint i = lid; i < BK * B2; i += threads) {
            uint row = i / B2;  // n offset
            uint col = i % B2;  // vci offset
            uint n_idx = bn + row;
            uint vci_idx = vci_base + col;
            if (n_idx < N && vci_idx < VCi) {
                uint v_idx = vci_idx / Ci;
                uint ci_idx = vci_idx % Ci;
                uint nb = neighbor[n_idx * V + v_idx];
                smem_b[row * B2 + col] = (nb != SENTINEL) ?
                    input[nb * Ci + ci_idx] : 0.0f;
            } else {
                smem_b[row * B2 + col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // simdgroup_matrix multiply
        simdgroup_matrix<float, 8, 8> a_mat;
        for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
            for (uint tn = 0; tn < TILES_N; tn++) {
                simdgroup_matrix<float, 8, 8> b_mat;
                simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results to shared memory then global
    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint co = co_base + row;
        uint vci = vci_base + col;
        if (co < Co && vci < VCi) {
            uint v_idx = vci / Ci;
            uint ci_idx = vci % Ci;
            grad_weight[(co * V + v_idx) * Ci + ci_idx] = smem_a[row * B2 + col];
        }
    }
}

// ============================================================================
// Half-precision and bfloat16 specializations.
//
// Same algorithm and tiling as the float32 kernels above. Buffers and tile
// shared memory are narrowed to ELEM_T (half or bfloat). Accumulators stay in
// float32 — the hardware simdgroup matmul supports half/bfloat on the inputs
// and float on the accumulator, which is the recommended mixed-precision mode
// on Apple Silicon. Loads/stores convert to/from float at the tile boundary.
//
// Shared memory layout for half/bfloat variants:
//   smem_a[B1][BK] | smem_b[BK][B2] | smem_nb[B1]  (fwd / bwd_input)
//   smem_a[B1][BK] | smem_b[BK][B2]                (bwd_weight)
//
// For the store-back-through-shared path we need a second float scratchpad for
// the accumulator tiles since ELEM_T storage would lose precision between the
// simdgroup_store and the coordinate-bounded write-out. To keep shared memory
// within the same 32KB envelope, we reuse smem_a's float view — total byte
// count is (B1*BK + BK*B2) * sizeof(ELEM_T) + B1*sizeof(uint32_t)
// + (B1*B2)*sizeof(float). For half this is 64*32*2 + 32*64*2 + 64*4 + 64*64*4
// = 4096 + 4096 + 256 + 16384 = 24832 bytes ≈ 24.2KB, well under the 32KB
// threadgroup limit on Apple Silicon.
// ============================================================================

#define SPCONV_FWD_KERNEL(NAME, ELEM_T)                                                   \
kernel void NAME(                                                                          \
    const device ELEM_T* input        [[buffer(0)]],                                       \
    const device ELEM_T* weight       [[buffer(1)]],                                       \
    const device ELEM_T* bias         [[buffer(2)]],                                       \
    const device uint*   neighbor     [[buffer(3)]],                                       \
    device       ELEM_T* output       [[buffer(4)]],                                       \
    constant     uint&   N            [[buffer(5)]],                                       \
    constant     uint&   Co           [[buffer(6)]],                                       \
    constant     uint&   Ci           [[buffer(7)]],                                       \
    constant     uint&   V            [[buffer(8)]],                                       \
    constant     uint&   has_bias     [[buffer(9)]],                                       \
    threadgroup  uchar*  smem_raw     [[threadgroup(0)]],                                  \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint n_base  = gid.x * B1;                                                             \
    uint co_base = gid.y * B2;                                                             \
    uint threads = GEMM_THREADS;                                                           \
                                                                                           \
    threadgroup ELEM_T* smem_a  = (threadgroup ELEM_T*)smem_raw;                           \
    threadgroup ELEM_T* smem_b  = smem_a + B1 * BK;                                        \
    threadgroup uint*   smem_nb = (threadgroup uint*)(smem_b + BK * B2);                   \
    threadgroup atomic_uint* smem_any_valid = (threadgroup atomic_uint*)(smem_nb + B1);    \
    threadgroup float*  smem_out = (threadgroup float*)(smem_any_valid + 1);               \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N];                                            \
    for (uint i = 0; i < TILES_N; i++)                                                     \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    for (uint v = 0; v < V; v++) {                                                         \
        if (lid == 0) {                                                                    \
            atomic_store_explicit(smem_any_valid, 0u, memory_order_relaxed);               \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        for (uint i = lid; i < B1; i += threads) {                                         \
            uint n = n_base + i;                                                           \
            uint nb = (n < N) ? neighbor[n * V + v] : SENTINEL;                            \
            smem_nb[i] = nb;                                                               \
            if (nb != SENTINEL) {                                                          \
                atomic_fetch_or_explicit(smem_any_valid, 1u, memory_order_relaxed);        \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        if (atomic_load_explicit(smem_any_valid, memory_order_relaxed) == 0) {             \
            continue;  /* No row in this block has a neighbor at v — skip entirely. */    \
        }                                                                                  \
                                                                                           \
        for (uint bk = 0; bk < Ci; bk += BK) {                                             \
            for (uint i = lid; i < B1 * BK; i += threads) {                                \
                uint row = i / BK;                                                         \
                uint col = i % BK;                                                         \
                uint ci_idx = bk + col;                                                    \
                uint nb_idx = smem_nb[row];                                                \
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?             \
                    input[nb_idx * Ci + ci_idx] : (ELEM_T)0;                               \
            }                                                                              \
            for (uint i = lid; i < BK * B2; i += threads) {                                \
                uint row = i / B2;                                                         \
                uint col = i % B2;                                                         \
                uint ci_idx = bk + row;                                                    \
                uint co_idx = co_base + col;                                               \
                smem_b[row * B2 + col] = (ci_idx < Ci && co_idx < Co) ?                    \
                    weight[(co_idx * V + v) * Ci + ci_idx] : (ELEM_T)0;                    \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                                                                                           \
            simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                          \
            for (uint k8 = 0; k8 < BK; k8 += 8) {                                          \
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                 \
                for (uint tn = 0; tn < TILES_N; tn++) {                                    \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                  \
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);                  \
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);         \
                }                                                                          \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    for (uint tn = 0; tn < TILES_N; tn++) {                                                \
        simdgroup_store(acc[tn], smem_out + simd_id * 8 * B2 + tn * 8, B2);                \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    for (uint i = lid; i < B1 * B2; i += threads) {                                        \
        uint row = i / B2;                                                                 \
        uint col = i % B2;                                                                 \
        uint n = n_base + row;                                                             \
        uint co = co_base + col;                                                           \
        if (n < N && co < Co) {                                                            \
            float val = smem_out[row * B2 + col];                                          \
            if (has_bias) val += (float)bias[co];                                          \
            output[n * Co + co] = (ELEM_T)val;                                             \
        }                                                                                  \
    }                                                                                      \
}

#define SPCONV_BWD_INPUT_KERNEL(NAME, ELEM_T)                                              \
kernel void NAME(                                                                          \
    const device ELEM_T* grad_output  [[buffer(0)]],                                       \
    const device ELEM_T* weight       [[buffer(1)]],                                       \
    const device uint*   neighbor     [[buffer(2)]],                                       \
    device       ELEM_T* grad_input   [[buffer(3)]],                                       \
    constant     uint&   N            [[buffer(4)]],                                       \
    constant     uint&   Co           [[buffer(5)]],                                       \
    constant     uint&   Ci           [[buffer(6)]],                                       \
    constant     uint&   V            [[buffer(7)]],                                       \
    threadgroup  uchar*  smem_raw     [[threadgroup(0)]],                                  \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint n_base  = gid.x * B1;                                                             \
    uint ci_base = gid.y * B2;                                                             \
    uint threads = GEMM_THREADS;                                                           \
                                                                                           \
    threadgroup ELEM_T* smem_a  = (threadgroup ELEM_T*)smem_raw;                           \
    threadgroup ELEM_T* smem_b  = smem_a + B1 * BK;                                        \
    threadgroup uint*   smem_nb = (threadgroup uint*)(smem_b + BK * B2);                   \
    threadgroup atomic_uint* smem_any_valid = (threadgroup atomic_uint*)(smem_nb + B1);    \
    threadgroup float*  smem_out = (threadgroup float*)(smem_any_valid + 1);               \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N];                                            \
    for (uint i = 0; i < TILES_N; i++)                                                     \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    for (uint v = 0; v < V; v++) {                                                         \
        uint v_flip = V - 1 - v;                                                           \
        if (lid == 0) {                                                                    \
            atomic_store_explicit(smem_any_valid, 0u, memory_order_relaxed);               \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        for (uint i = lid; i < B1; i += threads) {                                         \
            uint n = n_base + i;                                                           \
            uint nb = (n < N) ? neighbor[n * V + v_flip] : SENTINEL;                       \
            smem_nb[i] = nb;                                                               \
            if (nb != SENTINEL) {                                                          \
                atomic_fetch_or_explicit(smem_any_valid, 1u, memory_order_relaxed);        \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        if (atomic_load_explicit(smem_any_valid, memory_order_relaxed) == 0) {             \
            continue;                                                                      \
        }                                                                                  \
                                                                                           \
        for (uint bk = 0; bk < Co; bk += BK) {                                             \
            for (uint i = lid; i < B1 * BK; i += threads) {                                \
                uint row = i / BK;                                                         \
                uint col = i % BK;                                                         \
                uint co_idx = bk + col;                                                    \
                uint nb_idx = smem_nb[row];                                                \
                smem_a[row * BK + col] = (nb_idx != SENTINEL && co_idx < Co) ?             \
                    grad_output[nb_idx * Co + co_idx] : (ELEM_T)0;                         \
            }                                                                              \
            for (uint i = lid; i < BK * B2; i += threads) {                                \
                uint row = i / B2;                                                         \
                uint col = i % B2;                                                         \
                uint co_idx = bk + row;                                                    \
                uint ci_idx = ci_base + col;                                               \
                smem_b[row * B2 + col] = (co_idx < Co && ci_idx < Ci) ?                    \
                    weight[(co_idx * V + v) * Ci + ci_idx] : (ELEM_T)0;                    \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                                                                                           \
            simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                          \
            for (uint k8 = 0; k8 < BK; k8 += 8) {                                          \
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                 \
                for (uint tn = 0; tn < TILES_N; tn++) {                                    \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                  \
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);                  \
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);         \
                }                                                                          \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    for (uint tn = 0; tn < TILES_N; tn++) {                                                \
        simdgroup_store(acc[tn], smem_out + simd_id * 8 * B2 + tn * 8, B2);                \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    for (uint i = lid; i < B1 * B2; i += threads) {                                        \
        uint row = i / B2;                                                                 \
        uint col = i % B2;                                                                 \
        uint n = n_base + row;                                                             \
        uint ci = ci_base + col;                                                           \
        if (n < N && ci < Ci) {                                                            \
            grad_input[n * Ci + ci] = (ELEM_T)smem_out[row * B2 + col];                    \
        }                                                                                  \
    }                                                                                      \
}

#define SPCONV_BWD_WEIGHT_KERNEL(NAME, ELEM_T)                                             \
kernel void NAME(                                                                          \
    const device ELEM_T* grad_output  [[buffer(0)]],                                       \
    const device ELEM_T* input        [[buffer(1)]],                                       \
    const device uint*   neighbor     [[buffer(2)]],                                       \
    device       ELEM_T* grad_weight  [[buffer(3)]],                                       \
    constant     uint&   N            [[buffer(4)]],                                       \
    constant     uint&   Co           [[buffer(5)]],                                       \
    constant     uint&   Ci           [[buffer(6)]],                                       \
    constant     uint&   V            [[buffer(7)]],                                       \
    threadgroup  uchar*  smem_raw     [[threadgroup(0)]],                                  \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint co_base = gid.x * B1;                                                             \
    uint vci_base = gid.y * B2;                                                            \
    uint threads = GEMM_THREADS;                                                           \
    uint VCi = V * Ci;                                                                     \
                                                                                           \
    threadgroup ELEM_T* smem_a  = (threadgroup ELEM_T*)smem_raw;                           \
    threadgroup ELEM_T* smem_b  = smem_a + B1 * BK;                                        \
    threadgroup float*  smem_out = (threadgroup float*)(smem_b + BK * B2);                 \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N];                                            \
    for (uint i = 0; i < TILES_N; i++)                                                     \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    for (uint bn = 0; bn < N; bn += BK) {                                                  \
        for (uint i = lid; i < B1 * BK; i += threads) {                                    \
            uint row = i / BK;                                                             \
            uint col = i % BK;                                                             \
            uint co_idx = co_base + row;                                                   \
            uint n_idx = bn + col;                                                         \
            smem_a[row * BK + col] = (co_idx < Co && n_idx < N) ?                          \
                grad_output[n_idx * Co + co_idx] : (ELEM_T)0;                              \
        }                                                                                  \
        for (uint i = lid; i < BK * B2; i += threads) {                                    \
            uint row = i / B2;                                                             \
            uint col = i % B2;                                                             \
            uint n_idx = bn + row;                                                         \
            uint vci_idx = vci_base + col;                                                 \
            if (n_idx < N && vci_idx < VCi) {                                              \
                uint v_idx = vci_idx / Ci;                                                 \
                uint ci_idx = vci_idx % Ci;                                                \
                uint nb = neighbor[n_idx * V + v_idx];                                     \
                smem_b[row * B2 + col] = (nb != SENTINEL) ?                                \
                    input[nb * Ci + ci_idx] : (ELEM_T)0;                                   \
            } else {                                                                       \
                smem_b[row * B2 + col] = (ELEM_T)0;                                        \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                              \
        for (uint k8 = 0; k8 < BK; k8 += 8) {                                              \
            simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                     \
            for (uint tn = 0; tn < TILES_N; tn++) {                                        \
                simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                      \
                simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);                      \
                simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);             \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    }                                                                                      \
                                                                                           \
    for (uint tn = 0; tn < TILES_N; tn++) {                                                \
        simdgroup_store(acc[tn], smem_out + simd_id * 8 * B2 + tn * 8, B2);                \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    for (uint i = lid; i < B1 * B2; i += threads) {                                        \
        uint row = i / B2;                                                                 \
        uint col = i % B2;                                                                 \
        uint co = co_base + row;                                                           \
        uint vci = vci_base + col;                                                         \
        if (co < Co && vci < VCi) {                                                        \
            uint v_idx = vci / Ci;                                                         \
            uint ci_idx = vci % Ci;                                                        \
            grad_weight[(co * V + v_idx) * Ci + ci_idx] = (ELEM_T)smem_out[row * B2 + col];\
        }                                                                                  \
    }                                                                                      \
}

SPCONV_FWD_KERNEL(spconv_fwd_implicit_gemm_half, half)
SPCONV_FWD_KERNEL(spconv_fwd_implicit_gemm_bfloat, bfloat)

SPCONV_BWD_INPUT_KERNEL(spconv_bwd_input_implicit_gemm_half, half)
SPCONV_BWD_INPUT_KERNEL(spconv_bwd_input_implicit_gemm_bfloat, bfloat)

SPCONV_BWD_WEIGHT_KERNEL(spconv_bwd_weight_implicit_gemm_half, half)
SPCONV_BWD_WEIGHT_KERNEL(spconv_bwd_weight_implicit_gemm_bfloat, bfloat)

// ============================================================================
// Masked implicit GEMM forward — iterates only the valid V positions per
// n-block using the precomputed sorted_idx / valid_kernel / valid_kernel_seg
// machinery. Same simdgroup tiling as the dense kernel; the wins are:
//   - V loop bound = valid_kernel_seg[n_block+1] - valid_kernel_seg[n_block]
//     (often << V on sparse data)
//   - sorted_idx groups rows that share their valid V positions, so each block
//     wastes fewer iterations on rows that don't need them.
//
// Output is written at sorted_idx[row] positions, NOT at the dense block row,
// so we read sorted_idx into smem once and reuse across the V loop and the
// store-back loop.
//
// Smem layout (half/bfloat):
//   smem_a[B1][BK] | smem_b[BK][B2] | smem_nb[B1] | smem_sorted[B1] | smem_out[B1][B2]
//   = 4096 + 4096 + 256 + 256 + 16384 = 25088 bytes
//
// Smem layout (float32, output reuses smem_a+smem_b region):
//   smem_a+smem_b[16384 bytes] | smem_nb[256] | smem_sorted[256] = 16896 bytes
// ============================================================================

#define SPCONV_FWD_MASKED_KERNEL(NAME, ELEM_T)                                             \
kernel void NAME(                                                                          \
    const device ELEM_T* input            [[buffer(0)]],                                   \
    const device ELEM_T* weight           [[buffer(1)]],                                   \
    const device ELEM_T* bias             [[buffer(2)]],                                   \
    const device uint*   neighbor         [[buffer(3)]],                                   \
    const device int*    sorted_idx       [[buffer(4)]],                                   \
    const device int*    valid_kernel     [[buffer(5)]],                                   \
    const device int*    valid_kernel_seg [[buffer(6)]],                                   \
    device       ELEM_T* output           [[buffer(7)]],                                   \
    constant     uint&   N                [[buffer(8)]],                                   \
    constant     uint&   Co               [[buffer(9)]],                                   \
    constant     uint&   Ci               [[buffer(10)]],                                  \
    constant     uint&   V                [[buffer(11)]],                                  \
    constant     uint&   has_bias         [[buffer(12)]],                                  \
    threadgroup  uchar*  smem_raw         [[threadgroup(0)]],                              \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint n_block_idx = gid.x;                                                              \
    uint n_base   = n_block_idx * B1;                                                      \
    uint co_base  = gid.y * B2;                                                            \
    uint threads  = GEMM_THREADS;                                                          \
                                                                                           \
    threadgroup ELEM_T* smem_a    = (threadgroup ELEM_T*)smem_raw;                         \
    threadgroup ELEM_T* smem_b    = smem_a + B1 * BK;                                      \
    threadgroup uint*   smem_nb   = (threadgroup uint*)(smem_b + BK * B2);                 \
    threadgroup int*    smem_sorted = (threadgroup int*)(smem_nb + B1);                    \
    threadgroup float*  smem_out  = (threadgroup float*)(smem_sorted + B1);                \
                                                                                           \
    /* Load sorted row indices for this n-block once. -1 marks past-end rows. */          \
    for (uint i = lid; i < B1; i += threads) {                                             \
        uint n = n_base + i;                                                               \
        smem_sorted[i] = (n < N) ? sorted_idx[n] : -1;                                     \
    }                                                                                      \
                                                                                           \
    uint vps_start = (uint)valid_kernel_seg[n_block_idx];                                  \
    uint vps_end   = (uint)valid_kernel_seg[n_block_idx + 1];                              \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N];                                            \
    for (uint i = 0; i < TILES_N; i++)                                                     \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);  /* smem_sorted ready */              \
                                                                                           \
    for (uint vps_i = vps_start; vps_i < vps_end; vps_i++) {                               \
        uint v = (uint)valid_kernel[vps_i];                                                \
                                                                                           \
        /* Gather neighbor[sorted_idx[row], v] into smem_nb. */                            \
        for (uint i = lid; i < B1; i += threads) {                                         \
            int sn = smem_sorted[i];                                                       \
            smem_nb[i] = (sn >= 0) ? neighbor[(uint)sn * V + v] : SENTINEL;                \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        for (uint bk = 0; bk < Ci; bk += BK) {                                             \
            for (uint i = lid; i < B1 * BK; i += threads) {                                \
                uint row = i / BK;                                                         \
                uint col = i % BK;                                                         \
                uint ci_idx = bk + col;                                                    \
                uint nb_idx = smem_nb[row];                                                \
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?             \
                    input[nb_idx * Ci + ci_idx] : (ELEM_T)0;                               \
            }                                                                              \
            for (uint i = lid; i < BK * B2; i += threads) {                                \
                uint row = i / B2;                                                         \
                uint col = i % B2;                                                         \
                uint ci_idx = bk + row;                                                    \
                uint co_idx = co_base + col;                                               \
                smem_b[row * B2 + col] = (ci_idx < Ci && co_idx < Co) ?                    \
                    weight[(co_idx * V + v) * Ci + ci_idx] : (ELEM_T)0;                    \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                                                                                           \
            simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                          \
            for (uint k8 = 0; k8 < BK; k8 += 8) {                                          \
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                 \
                for (uint tn = 0; tn < TILES_N; tn++) {                                    \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                  \
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);                  \
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);         \
                }                                                                          \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    for (uint tn = 0; tn < TILES_N; tn++) {                                                \
        simdgroup_store(acc[tn], smem_out + simd_id * 8 * B2 + tn * 8, B2);                \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    /* Scatter to original row positions via sorted_idx. */                                \
    for (uint i = lid; i < B1 * B2; i += threads) {                                        \
        uint row = i / B2;                                                                 \
        uint col = i % B2;                                                                 \
        int sn = smem_sorted[row];                                                         \
        uint co = co_base + col;                                                           \
        if (sn >= 0 && co < Co) {                                                          \
            float val = smem_out[row * B2 + col];                                          \
            if (has_bias) val += (float)bias[co];                                          \
            output[(uint)sn * Co + co] = (ELEM_T)val;                                      \
        }                                                                                  \
    }                                                                                      \
}

// fp32 specialization: output reuses the smem_a+smem_b region (16384 bytes of
// float scratch, > the 64*64*4 = 16384 bytes the simdgroup_store writes).
kernel void spconv_fwd_masked_implicit_gemm(
    const device float* input            [[buffer(0)]],
    const device float* weight           [[buffer(1)]],
    const device float* bias             [[buffer(2)]],
    const device uint*  neighbor         [[buffer(3)]],
    const device int*   sorted_idx       [[buffer(4)]],
    const device int*   valid_kernel     [[buffer(5)]],
    const device int*   valid_kernel_seg [[buffer(6)]],
    device       float* output           [[buffer(7)]],
    constant     uint&  N                [[buffer(8)]],
    constant     uint&  Co               [[buffer(9)]],
    constant     uint&  Ci               [[buffer(10)]],
    constant     uint&  V                [[buffer(11)]],
    constant     uint&  has_bias         [[buffer(12)]],
    threadgroup  float* smem             [[threadgroup(0)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  simd_id  [[simdgroup_index_in_threadgroup]],
    uint  lane_id  [[thread_index_in_simdgroup]]
) {
    uint n_block_idx = gid.x;
    uint n_base  = n_block_idx * B1;
    uint co_base = gid.y * B2;
    uint threads = GEMM_THREADS;

    threadgroup float* smem_a       = smem;
    threadgroup float* smem_b       = smem + B1 * BK;
    threadgroup uint*  smem_nb      = (threadgroup uint*)(smem + B1 * BK + BK * B2);
    threadgroup int*   smem_sorted  = (threadgroup int*)(smem_nb + B1);

    for (uint i = lid; i < B1; i += threads) {
        uint n = n_base + i;
        smem_sorted[i] = (n < N) ? sorted_idx[n] : -1;
    }

    uint vps_start = (uint)valid_kernel_seg[n_block_idx];
    uint vps_end   = (uint)valid_kernel_seg[n_block_idx + 1];

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint vps_i = vps_start; vps_i < vps_end; vps_i++) {
        uint v = (uint)valid_kernel[vps_i];

        for (uint i = lid; i < B1; i += threads) {
            int sn = smem_sorted[i];
            smem_nb[i] = (sn >= 0) ? neighbor[(uint)sn * V + v] : SENTINEL;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bk = 0; bk < Ci; bk += BK) {
            for (uint i = lid; i < B1 * BK; i += threads) {
                uint row = i / BK;
                uint col = i % BK;
                uint ci_idx = bk + col;
                uint nb_idx = smem_nb[row];
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?
                    input[nb_idx * Ci + ci_idx] : 0.0f;
            }
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;
                uint col = i % B2;
                uint ci_idx = bk + row;
                uint co_idx = co_base + col;
                smem_b[row * B2 + col] = (ci_idx < Ci && co_idx < Co) ?
                    weight[(co_idx * V + v) * Ci + ci_idx] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> a_mat;
            for (uint k8 = 0; k8 < BK; k8 += 8) {
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
                for (uint tn = 0; tn < TILES_N; tn++) {
                    simdgroup_matrix<float, 8, 8> b_mat;
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        int sn = smem_sorted[row];
        uint co = co_base + col;
        if (sn >= 0 && co < Co) {
            float val = smem_a[row * B2 + col];
            if (has_bias) val += bias[co];
            output[(uint)sn * Co + co] = val;
        }
    }
}

SPCONV_FWD_MASKED_KERNEL(spconv_fwd_masked_implicit_gemm_half, half)
SPCONV_FWD_MASKED_KERNEL(spconv_fwd_masked_implicit_gemm_bfloat, bfloat)

// ============================================================================
// Masked implicit GEMM backward-input — uses the same precomputed
// sorted_idx / valid_kernel / valid_kernel_seg as the forward masked kernel.
// Algebra (from the Triton reference):
//   grad_input[n_sorted, ci] = sum_{v in valid_kernel[n_block]} sum_co
//        weight[co, V-1-v, ci] * grad_output[neighbor[n_sorted, v], co]
// Substituting u = V-1-v gives the dense formulation
//   grad_input[n, ci] = sum_u sum_co weight[co, u, ci] * grad_output[neighbor[n, V-1-u], co]
// — so the same valid_kernel set works for both directions.
//
// Smem layout (half/bfloat):
//   smem_a[B1*BK] | smem_b[BK*B2] | smem_nb[B1] | smem_sorted[B1] | smem_out[B1*B2]
//   = 4096 + 4096 + 256 + 256 + 16384 = 25088 bytes
//
// Smem layout (float32, output reuses smem_a region):
//   smem_a+smem_b[16384] | smem_nb[256] | smem_sorted[256] = 16896 bytes
// ============================================================================

#define SPCONV_BWD_INPUT_MASKED_KERNEL(NAME, ELEM_T)                                       \
kernel void NAME(                                                                          \
    const device ELEM_T* grad_output      [[buffer(0)]],                                   \
    const device ELEM_T* weight           [[buffer(1)]],                                   \
    const device uint*   neighbor         [[buffer(2)]],                                   \
    const device int*    sorted_idx       [[buffer(3)]],                                   \
    const device int*    valid_kernel     [[buffer(4)]],                                   \
    const device int*    valid_kernel_seg [[buffer(5)]],                                   \
    device       ELEM_T* grad_input       [[buffer(6)]],                                   \
    constant     uint&   N                [[buffer(7)]],                                   \
    constant     uint&   Co               [[buffer(8)]],                                   \
    constant     uint&   Ci               [[buffer(9)]],                                   \
    constant     uint&   V                [[buffer(10)]],                                  \
    threadgroup  uchar*  smem_raw         [[threadgroup(0)]],                              \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint n_block_idx = gid.x;                                                              \
    uint n_base   = n_block_idx * B1;                                                      \
    uint ci_base  = gid.y * B2;                                                            \
    uint threads  = GEMM_THREADS;                                                          \
                                                                                           \
    threadgroup ELEM_T* smem_a    = (threadgroup ELEM_T*)smem_raw;                         \
    threadgroup ELEM_T* smem_b    = smem_a + B1 * BK;                                      \
    threadgroup uint*   smem_nb   = (threadgroup uint*)(smem_b + BK * B2);                 \
    threadgroup int*    smem_sorted = (threadgroup int*)(smem_nb + B1);                    \
    threadgroup float*  smem_out  = (threadgroup float*)(smem_sorted + B1);                \
                                                                                           \
    for (uint i = lid; i < B1; i += threads) {                                             \
        uint n = n_base + i;                                                               \
        smem_sorted[i] = (n < N) ? sorted_idx[n] : -1;                                     \
    }                                                                                      \
                                                                                           \
    uint vps_start = (uint)valid_kernel_seg[n_block_idx];                                  \
    uint vps_end   = (uint)valid_kernel_seg[n_block_idx + 1];                              \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N];                                            \
    for (uint i = 0; i < TILES_N; i++)                                                     \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);  /* smem_sorted ready */              \
                                                                                           \
    for (uint vps_i = vps_start; vps_i < vps_end; vps_i++) {                               \
        uint v = (uint)valid_kernel[vps_i];                                                \
        uint v_w = V - 1 - v;  /* weight uses flipped V (see header comment) */            \
                                                                                           \
        for (uint i = lid; i < B1; i += threads) {                                         \
            int sn = smem_sorted[i];                                                       \
            smem_nb[i] = (sn >= 0) ? neighbor[(uint)sn * V + v] : SENTINEL;                \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        for (uint bk = 0; bk < Co; bk += BK) {                                             \
            for (uint i = lid; i < B1 * BK; i += threads) {                                \
                uint row = i / BK;                                                         \
                uint col = i % BK;                                                         \
                uint co_idx = bk + col;                                                    \
                uint nb_idx = smem_nb[row];                                                \
                smem_a[row * BK + col] = (nb_idx != SENTINEL && co_idx < Co) ?             \
                    grad_output[nb_idx * Co + co_idx] : (ELEM_T)0;                         \
            }                                                                              \
            for (uint i = lid; i < BK * B2; i += threads) {                                \
                uint row = i / B2;                                                         \
                uint col = i % B2;                                                         \
                uint co_idx = bk + row;                                                    \
                uint ci_idx = ci_base + col;                                               \
                smem_b[row * B2 + col] = (co_idx < Co && ci_idx < Ci) ?                    \
                    weight[(co_idx * V + v_w) * Ci + ci_idx] : (ELEM_T)0;                  \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                                                                                           \
            simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                          \
            for (uint k8 = 0; k8 < BK; k8 += 8) {                                          \
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                 \
                for (uint tn = 0; tn < TILES_N; tn++) {                                    \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                  \
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);                  \
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);         \
                }                                                                          \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    for (uint tn = 0; tn < TILES_N; tn++) {                                                \
        simdgroup_store(acc[tn], smem_out + simd_id * 8 * B2 + tn * 8, B2);                \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    /* Scatter to original row positions via sorted_idx. */                                \
    for (uint i = lid; i < B1 * B2; i += threads) {                                        \
        uint row = i / B2;                                                                 \
        uint col = i % B2;                                                                 \
        int sn = smem_sorted[row];                                                         \
        uint ci = ci_base + col;                                                           \
        if (sn >= 0 && ci < Ci) {                                                          \
            grad_input[(uint)sn * Ci + ci] = (ELEM_T)smem_out[row * B2 + col];             \
        }                                                                                  \
    }                                                                                      \
}

// fp32 specialization: output reuses smem_a region (16384 bytes scratch >= 64*64*4).
kernel void spconv_bwd_input_masked_implicit_gemm(
    const device float* grad_output      [[buffer(0)]],
    const device float* weight           [[buffer(1)]],
    const device uint*  neighbor         [[buffer(2)]],
    const device int*   sorted_idx       [[buffer(3)]],
    const device int*   valid_kernel     [[buffer(4)]],
    const device int*   valid_kernel_seg [[buffer(5)]],
    device       float* grad_input       [[buffer(6)]],
    constant     uint&  N                [[buffer(7)]],
    constant     uint&  Co               [[buffer(8)]],
    constant     uint&  Ci               [[buffer(9)]],
    constant     uint&  V                [[buffer(10)]],
    threadgroup  float* smem             [[threadgroup(0)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  simd_id  [[simdgroup_index_in_threadgroup]],
    uint  lane_id  [[thread_index_in_simdgroup]]
) {
    uint n_block_idx = gid.x;
    uint n_base  = n_block_idx * B1;
    uint ci_base = gid.y * B2;
    uint threads = GEMM_THREADS;

    threadgroup float* smem_a       = smem;
    threadgroup float* smem_b       = smem + B1 * BK;
    threadgroup uint*  smem_nb      = (threadgroup uint*)(smem + B1 * BK + BK * B2);
    threadgroup int*   smem_sorted  = (threadgroup int*)(smem_nb + B1);

    for (uint i = lid; i < B1; i += threads) {
        uint n = n_base + i;
        smem_sorted[i] = (n < N) ? sorted_idx[n] : -1;
    }

    uint vps_start = (uint)valid_kernel_seg[n_block_idx];
    uint vps_end   = (uint)valid_kernel_seg[n_block_idx + 1];

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint vps_i = vps_start; vps_i < vps_end; vps_i++) {
        uint v = (uint)valid_kernel[vps_i];
        uint v_w = V - 1 - v;

        for (uint i = lid; i < B1; i += threads) {
            int sn = smem_sorted[i];
            smem_nb[i] = (sn >= 0) ? neighbor[(uint)sn * V + v] : SENTINEL;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bk = 0; bk < Co; bk += BK) {
            for (uint i = lid; i < B1 * BK; i += threads) {
                uint row = i / BK;
                uint col = i % BK;
                uint co_idx = bk + col;
                uint nb_idx = smem_nb[row];
                smem_a[row * BK + col] = (nb_idx != SENTINEL && co_idx < Co) ?
                    grad_output[nb_idx * Co + co_idx] : 0.0f;
            }
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;
                uint col = i % B2;
                uint co_idx = bk + row;
                uint ci_idx = ci_base + col;
                smem_b[row * B2 + col] = (co_idx < Co && ci_idx < Ci) ?
                    weight[(co_idx * V + v_w) * Ci + ci_idx] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> a_mat;
            for (uint k8 = 0; k8 < BK; k8 += 8) {
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
                for (uint tn = 0; tn < TILES_N; tn++) {
                    simdgroup_matrix<float, 8, 8> b_mat;
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        int sn = smem_sorted[row];
        uint ci = ci_base + col;
        if (sn >= 0 && ci < Ci) {
            grad_input[(uint)sn * Ci + ci] = smem_a[row * B2 + col];
        }
    }
}

SPCONV_BWD_INPUT_MASKED_KERNEL(spconv_bwd_input_masked_implicit_gemm_half, half)
SPCONV_BWD_INPUT_MASKED_KERNEL(spconv_bwd_input_masked_implicit_gemm_bfloat, bfloat)

// ============================================================================
// Masked implicit GEMM backward-weight — uses precomputed
// valid_signal_i / valid_signal_o / valid_signal_seg from
// neighbor_map_post_process_for_masked_implicit_gemm_1. valid_signal_seg is
// length V+1: for each kernel offset v, the segment lists (n_input, n_output)
// pairs where neighbor[n_output, v] = n_input. So:
//   grad_weight[co, v, ci] = sum_{(ni,no) in seg[v]}
//        grad_output[no, co] * input[ni, ci]
//
// Grid: (cdiv(Co, B1), cdiv(Ci, B2), V) — one threadgroup per (co_tile, ci_tile, v).
//
// Smem layout (half/bfloat):
//   smem_a[B1*BK] (grad_output^T) | smem_b[BK*B2] (input gathered)
//   | smem_si[BK] | smem_so[BK] | smem_out[B1*B2]
//   = 4096 + 4096 + 128 + 128 + 16384 = 24832 bytes
//
// Smem layout (float32, output reuses smem_a):
//   smem_a+smem_b[16384] | smem_si[128] | smem_so[128] = 16640 bytes
// ============================================================================

#define SPCONV_BWD_WEIGHT_MASKED_KERNEL(NAME, ELEM_T)                                      \
kernel void NAME(                                                                          \
    const device ELEM_T* grad_output      [[buffer(0)]],                                   \
    const device ELEM_T* input            [[buffer(1)]],                                   \
    const device uint*   valid_signal_i   [[buffer(2)]],                                   \
    const device uint*   valid_signal_o   [[buffer(3)]],                                   \
    const device uint*   valid_signal_seg [[buffer(4)]],                                   \
    device       ELEM_T* grad_weight      [[buffer(5)]],                                   \
    constant     uint&   N                [[buffer(6)]],                                   \
    constant     uint&   Co               [[buffer(7)]],                                   \
    constant     uint&   Ci               [[buffer(8)]],                                   \
    constant     uint&   V                [[buffer(9)]],                                   \
    threadgroup  uchar*  smem_raw         [[threadgroup(0)]],                              \
    uint3 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint co_base = gid.x * B1;                                                             \
    uint ci_base = gid.y * B2;                                                             \
    uint v       = gid.z;                                                                  \
    uint threads = GEMM_THREADS;                                                           \
                                                                                           \
    threadgroup ELEM_T* smem_a   = (threadgroup ELEM_T*)smem_raw;                          \
    threadgroup ELEM_T* smem_b   = smem_a + B1 * BK;                                       \
    threadgroup uint*   smem_si  = (threadgroup uint*)(smem_b + BK * B2);                  \
    threadgroup uint*   smem_so  = smem_si + BK;                                           \
    threadgroup float*  smem_out = (threadgroup float*)(smem_so + BK);                     \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N];                                            \
    for (uint i = 0; i < TILES_N; i++)                                                     \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    uint seg_start = valid_signal_seg[v];                                                  \
    uint seg_end   = valid_signal_seg[v + 1];                                              \
    uint seg_len   = seg_end - seg_start;                                                  \
                                                                                           \
    for (uint bk = 0; bk < seg_len; bk += BK) {                                            \
        /* Load BK pairs of (n_input, n_output) for this k-tile. */                        \
        for (uint i = lid; i < BK; i += threads) {                                         \
            uint k = bk + i;                                                               \
            if (k < seg_len) {                                                             \
                smem_si[i] = valid_signal_i[seg_start + k];                                \
                smem_so[i] = valid_signal_o[seg_start + k];                                \
            } else {                                                                       \
                smem_si[i] = SENTINEL;                                                     \
                smem_so[i] = SENTINEL;                                                     \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        /* smem_a[B1][BK] = grad_output^T[co, n_o] (rows=co, cols=k) */                    \
        for (uint i = lid; i < B1 * BK; i += threads) {                                    \
            uint row = i / BK;  /* co offset */                                            \
            uint col = i % BK;  /* k offset */                                             \
            uint co_idx = co_base + row;                                                   \
            uint no = smem_so[col];                                                        \
            smem_a[row * BK + col] = (co_idx < Co && no != SENTINEL) ?                     \
                grad_output[no * Co + co_idx] : (ELEM_T)0;                                 \
        }                                                                                  \
                                                                                           \
        /* smem_b[BK][B2] = input[n_i, ci] (rows=k, cols=ci) */                            \
        for (uint i = lid; i < BK * B2; i += threads) {                                    \
            uint row = i / B2;  /* k offset */                                             \
            uint col = i % B2;  /* ci offset */                                            \
            uint ci_idx = ci_base + col;                                                   \
            uint ni = smem_si[row];                                                        \
            smem_b[row * B2 + col] = (ci_idx < Ci && ni != SENTINEL) ?                     \
                input[ni * Ci + ci_idx] : (ELEM_T)0;                                       \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                              \
        for (uint k8 = 0; k8 < BK; k8 += 8) {                                              \
            simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                     \
            for (uint tn = 0; tn < TILES_N; tn++) {                                        \
                simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                      \
                simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);                      \
                simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);             \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    }                                                                                      \
                                                                                           \
    for (uint tn = 0; tn < TILES_N; tn++) {                                                \
        simdgroup_store(acc[tn], smem_out + simd_id * 8 * B2 + tn * 8, B2);                \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    for (uint i = lid; i < B1 * B2; i += threads) {                                        \
        uint row = i / B2;                                                                 \
        uint col = i % B2;                                                                 \
        uint co  = co_base + row;                                                          \
        uint ci  = ci_base + col;                                                          \
        if (co < Co && ci < Ci) {                                                          \
            grad_weight[(co * V + v) * Ci + ci] = (ELEM_T)smem_out[row * B2 + col];        \
        }                                                                                  \
    }                                                                                      \
}

// fp32 specialization: output reuses smem_a (16384 bytes).
kernel void spconv_bwd_weight_masked_implicit_gemm(
    const device float* grad_output      [[buffer(0)]],
    const device float* input            [[buffer(1)]],
    const device uint*  valid_signal_i   [[buffer(2)]],
    const device uint*  valid_signal_o   [[buffer(3)]],
    const device uint*  valid_signal_seg [[buffer(4)]],
    device       float* grad_weight      [[buffer(5)]],
    constant     uint&  N                [[buffer(6)]],
    constant     uint&  Co               [[buffer(7)]],
    constant     uint&  Ci               [[buffer(8)]],
    constant     uint&  V                [[buffer(9)]],
    threadgroup  float* smem             [[threadgroup(0)]],
    uint3 gid   [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  simd_id  [[simdgroup_index_in_threadgroup]],
    uint  lane_id  [[thread_index_in_simdgroup]]
) {
    uint co_base = gid.x * B1;
    uint ci_base = gid.y * B2;
    uint v       = gid.z;
    uint threads = GEMM_THREADS;

    threadgroup float* smem_a  = smem;
    threadgroup float* smem_b  = smem + B1 * BK;
    threadgroup uint*  smem_si = (threadgroup uint*)(smem + B1 * BK + BK * B2);
    threadgroup uint*  smem_so = smem_si + BK;

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint seg_start = valid_signal_seg[v];
    uint seg_end   = valid_signal_seg[v + 1];
    uint seg_len   = seg_end - seg_start;

    for (uint bk = 0; bk < seg_len; bk += BK) {
        for (uint i = lid; i < BK; i += threads) {
            uint k = bk + i;
            if (k < seg_len) {
                smem_si[i] = valid_signal_i[seg_start + k];
                smem_so[i] = valid_signal_o[seg_start + k];
            } else {
                smem_si[i] = SENTINEL;
                smem_so[i] = SENTINEL;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < B1 * BK; i += threads) {
            uint row = i / BK;
            uint col = i % BK;
            uint co_idx = co_base + row;
            uint no = smem_so[col];
            smem_a[row * BK + col] = (co_idx < Co && no != SENTINEL) ?
                grad_output[no * Co + co_idx] : 0.0f;
        }
        for (uint i = lid; i < BK * B2; i += threads) {
            uint row = i / B2;
            uint col = i % B2;
            uint ci_idx = ci_base + col;
            uint ni = smem_si[row];
            smem_b[row * B2 + col] = (ci_idx < Ci && ni != SENTINEL) ?
                input[ni * Ci + ci_idx] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a_mat;
        for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
            for (uint tn = 0; tn < TILES_N; tn++) {
                simdgroup_matrix<float, 8, 8> b_mat;
                simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint co  = co_base + row;
        uint ci  = ci_base + col;
        if (co < Co && ci < Ci) {
            grad_weight[(co * V + v) * Ci + ci] = smem_a[row * B2 + col];
        }
    }
}

SPCONV_BWD_WEIGHT_MASKED_KERNEL(spconv_bwd_weight_masked_implicit_gemm_half, half)
SPCONV_BWD_WEIGHT_MASKED_KERNEL(spconv_bwd_weight_masked_implicit_gemm_bfloat, bfloat)

// ============================================================================
// Wide-tile (B2=128) variants — for ch >= 128 shapes.
//
// The 64x64 tile underutilizes wide outputs at ch=128/256: each output tile
// is only 64 cols wide, so we pay launch overhead per 64-col strip.
// B2=128 doubles the output cols per threadgroup, halving the number of
// dispatched groups along the Co axis.
//
// The constraint is threadgroup memory: a B1*B2 fp32 scratch for the
// accumulator store-back would be 64*128*4 = 32KB on its own, blowing the
// limit. We stage output in two passes — accumulators[0..7] cover cols
// 0..63, accumulators[8..15] cover cols 64..127. After the K loop:
//   pass 1: simdgroup_store acc[0..7] → smem_out (16KB) → global cols 0..63
//   pass 2: simdgroup_store acc[8..15] → smem_out (reused) → global cols 64..127
//
// fp32 tile128 is omitted — production trellis2 is fp16/bf16 and the fp32
// kernel is exercised on CPU where launch overhead is irrelevant.
// ============================================================================

#define B2_T128 128
#define TILES_N_T128 (B2_T128 / 8)        // 16 accumulators per simdgroup
#define TILES_N_HALF (TILES_N_T128 / 2)   // 8 accumulators per output pass
#define HALF_COLS    (TILES_N_HALF * 8)   // 64 cols per output pass

#define SPCONV_FWD_KERNEL_T128(NAME, ELEM_T)                                               \
kernel void NAME(                                                                          \
    const device ELEM_T* input        [[buffer(0)]],                                       \
    const device ELEM_T* weight       [[buffer(1)]],                                       \
    const device ELEM_T* bias         [[buffer(2)]],                                       \
    const device uint*   neighbor     [[buffer(3)]],                                       \
    device       ELEM_T* output       [[buffer(4)]],                                       \
    constant     uint&   N            [[buffer(5)]],                                       \
    constant     uint&   Co           [[buffer(6)]],                                       \
    constant     uint&   Ci           [[buffer(7)]],                                       \
    constant     uint&   V            [[buffer(8)]],                                       \
    constant     uint&   has_bias     [[buffer(9)]],                                       \
    threadgroup  uchar*  smem_raw     [[threadgroup(0)]],                                  \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint n_base  = gid.x * B1;                                                             \
    uint co_base = gid.y * B2_T128;                                                        \
    uint threads = GEMM_THREADS;                                                           \
                                                                                           \
    threadgroup ELEM_T* smem_a      = (threadgroup ELEM_T*)smem_raw;                       \
    threadgroup ELEM_T* smem_b      = smem_a + B1 * BK;                                    \
    threadgroup uint*   smem_nb     = (threadgroup uint*)(smem_b + BK * B2_T128);          \
    threadgroup atomic_uint* smem_any_valid = (threadgroup atomic_uint*)(smem_nb + B1);    \
    threadgroup float*  smem_out    = (threadgroup float*)(smem_any_valid + 1);            \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N_T128];                                       \
    for (uint i = 0; i < TILES_N_T128; i++)                                                \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    for (uint v = 0; v < V; v++) {                                                         \
        if (lid == 0) {                                                                    \
            atomic_store_explicit(smem_any_valid, 0u, memory_order_relaxed);               \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        for (uint i = lid; i < B1; i += threads) {                                         \
            uint n = n_base + i;                                                           \
            uint nb = (n < N) ? neighbor[n * V + v] : SENTINEL;                            \
            smem_nb[i] = nb;                                                               \
            if (nb != SENTINEL) {                                                          \
                atomic_fetch_or_explicit(smem_any_valid, 1u, memory_order_relaxed);        \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        if (atomic_load_explicit(smem_any_valid, memory_order_relaxed) == 0) {             \
            continue;                                                                      \
        }                                                                                  \
                                                                                           \
        for (uint bk = 0; bk < Ci; bk += BK) {                                             \
            for (uint i = lid; i < B1 * BK; i += threads) {                                \
                uint row = i / BK;                                                         \
                uint col = i % BK;                                                         \
                uint ci_idx = bk + col;                                                    \
                uint nb_idx = smem_nb[row];                                                \
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?             \
                    input[nb_idx * Ci + ci_idx] : (ELEM_T)0;                               \
            }                                                                              \
            for (uint i = lid; i < BK * B2_T128; i += threads) {                           \
                uint row = i / B2_T128;                                                    \
                uint col = i % B2_T128;                                                    \
                uint ci_idx = bk + row;                                                    \
                uint co_idx = co_base + col;                                               \
                smem_b[row * B2_T128 + col] = (ci_idx < Ci && co_idx < Co) ?               \
                    weight[(co_idx * V + v) * Ci + ci_idx] : (ELEM_T)0;                    \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                                                                                           \
            simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                          \
            for (uint k8 = 0; k8 < BK; k8 += 8) {                                          \
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                 \
                for (uint tn = 0; tn < TILES_N_T128; tn++) {                               \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                  \
                    simdgroup_load(b_mat, smem_b + k8 * B2_T128 + tn * 8, B2_T128);        \
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);         \
                }                                                                          \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    /* Two-pass output staging: each pass writes HALF_COLS=64 columns through */           \
    /* the same smem_out scratch (16KB). */                                                \
    for (uint pass = 0; pass < 2; pass++) {                                                \
        uint pass_col_base = pass * HALF_COLS;                                             \
        for (uint tn = 0; tn < TILES_N_HALF; tn++) {                                       \
            simdgroup_store(acc[pass * TILES_N_HALF + tn],                                 \
                            smem_out + simd_id * 8 * HALF_COLS + tn * 8,                   \
                            HALF_COLS);                                                    \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        for (uint i = lid; i < B1 * HALF_COLS; i += threads) {                             \
            uint row = i / HALF_COLS;                                                      \
            uint col = i % HALF_COLS;                                                      \
            uint n = n_base + row;                                                         \
            uint co = co_base + pass_col_base + col;                                       \
            if (n < N && co < Co) {                                                        \
                float val = smem_out[row * HALF_COLS + col];                               \
                if (has_bias) val += (float)bias[co];                                      \
                output[n * Co + co] = (ELEM_T)val;                                         \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    }                                                                                      \
}

#define SPCONV_FWD_MASKED_KERNEL_T128(NAME, ELEM_T)                                        \
kernel void NAME(                                                                          \
    const device ELEM_T* input            [[buffer(0)]],                                   \
    const device ELEM_T* weight           [[buffer(1)]],                                   \
    const device ELEM_T* bias             [[buffer(2)]],                                   \
    const device uint*   neighbor         [[buffer(3)]],                                   \
    const device int*    sorted_idx       [[buffer(4)]],                                   \
    const device int*    valid_kernel     [[buffer(5)]],                                   \
    const device int*    valid_kernel_seg [[buffer(6)]],                                   \
    device       ELEM_T* output           [[buffer(7)]],                                   \
    constant     uint&   N                [[buffer(8)]],                                   \
    constant     uint&   Co               [[buffer(9)]],                                   \
    constant     uint&   Ci               [[buffer(10)]],                                  \
    constant     uint&   V                [[buffer(11)]],                                  \
    constant     uint&   has_bias         [[buffer(12)]],                                  \
    threadgroup  uchar*  smem_raw         [[threadgroup(0)]],                              \
    uint2 gid   [[threadgroup_position_in_grid]],                                          \
    uint  lid   [[thread_index_in_threadgroup]],                                           \
    uint  simd_id  [[simdgroup_index_in_threadgroup]],                                     \
    uint  lane_id  [[thread_index_in_simdgroup]]                                           \
) {                                                                                        \
    uint n_block_idx = gid.x;                                                              \
    uint n_base   = n_block_idx * B1;                                                      \
    uint co_base  = gid.y * B2_T128;                                                       \
    uint threads  = GEMM_THREADS;                                                          \
                                                                                           \
    threadgroup ELEM_T* smem_a      = (threadgroup ELEM_T*)smem_raw;                       \
    threadgroup ELEM_T* smem_b      = smem_a + B1 * BK;                                    \
    threadgroup uint*   smem_nb     = (threadgroup uint*)(smem_b + BK * B2_T128);          \
    threadgroup int*    smem_sorted = (threadgroup int*)(smem_nb + B1);                    \
    threadgroup float*  smem_out    = (threadgroup float*)(smem_sorted + B1);              \
                                                                                           \
    for (uint i = lid; i < B1; i += threads) {                                             \
        uint n = n_base + i;                                                               \
        smem_sorted[i] = (n < N) ? sorted_idx[n] : -1;                                     \
    }                                                                                      \
                                                                                           \
    uint vps_start = (uint)valid_kernel_seg[n_block_idx];                                  \
    uint vps_end   = (uint)valid_kernel_seg[n_block_idx + 1];                              \
                                                                                           \
    simdgroup_matrix<float, 8, 8> acc[TILES_N_T128];                                       \
    for (uint i = 0; i < TILES_N_T128; i++)                                                \
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);                                      \
                                                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                           \
    for (uint vps_i = vps_start; vps_i < vps_end; vps_i++) {                               \
        uint v = (uint)valid_kernel[vps_i];                                                \
                                                                                           \
        for (uint i = lid; i < B1; i += threads) {                                         \
            int sn = smem_sorted[i];                                                       \
            smem_nb[i] = (sn >= 0) ? neighbor[(uint)sn * V + v] : SENTINEL;                \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        for (uint bk = 0; bk < Ci; bk += BK) {                                             \
            for (uint i = lid; i < B1 * BK; i += threads) {                                \
                uint row = i / BK;                                                         \
                uint col = i % BK;                                                         \
                uint ci_idx = bk + col;                                                    \
                uint nb_idx = smem_nb[row];                                                \
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?             \
                    input[nb_idx * Ci + ci_idx] : (ELEM_T)0;                               \
            }                                                                              \
            for (uint i = lid; i < BK * B2_T128; i += threads) {                           \
                uint row = i / B2_T128;                                                    \
                uint col = i % B2_T128;                                                    \
                uint ci_idx = bk + row;                                                    \
                uint co_idx = co_base + col;                                               \
                smem_b[row * B2_T128 + col] = (ci_idx < Ci && co_idx < Co) ?               \
                    weight[(co_idx * V + v) * Ci + ci_idx] : (ELEM_T)0;                    \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                                                                                           \
            simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                          \
            for (uint k8 = 0; k8 < BK; k8 += 8) {                                          \
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);                 \
                for (uint tn = 0; tn < TILES_N_T128; tn++) {                               \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                  \
                    simdgroup_load(b_mat, smem_b + k8 * B2_T128 + tn * 8, B2_T128);        \
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);         \
                }                                                                          \
            }                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    for (uint pass = 0; pass < 2; pass++) {                                                \
        uint pass_col_base = pass * HALF_COLS;                                             \
        for (uint tn = 0; tn < TILES_N_HALF; tn++) {                                       \
            simdgroup_store(acc[pass * TILES_N_HALF + tn],                                 \
                            smem_out + simd_id * 8 * HALF_COLS + tn * 8,                   \
                            HALF_COLS);                                                    \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        for (uint i = lid; i < B1 * HALF_COLS; i += threads) {                             \
            uint row = i / HALF_COLS;                                                      \
            uint col = i % HALF_COLS;                                                      \
            int sn = smem_sorted[row];                                                     \
            uint co = co_base + pass_col_base + col;                                       \
            if (sn >= 0 && co < Co) {                                                      \
                float val = smem_out[row * HALF_COLS + col];                               \
                if (has_bias) val += (float)bias[co];                                      \
                output[(uint)sn * Co + co] = (ELEM_T)val;                                  \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    }                                                                                      \
}

SPCONV_FWD_KERNEL_T128(spconv_fwd_implicit_gemm_t128_half, half)
SPCONV_FWD_KERNEL_T128(spconv_fwd_implicit_gemm_t128_bfloat, bfloat)

SPCONV_FWD_MASKED_KERNEL_T128(spconv_fwd_masked_implicit_gemm_t128_half, half)
SPCONV_FWD_MASKED_KERNEL_T128(spconv_fwd_masked_implicit_gemm_t128_bfloat, bfloat)
