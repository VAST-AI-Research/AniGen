#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Flash-attention-v2 backward for variable-length sparse sequences.
//
// Two kernels, same simdgroup-matmul style as fwd in sparse_attn_tiled.metal.
// Each kernel mirrors the 3-pass structure of the naive bwd (m/l → D → dQ),
// but replaces every O(BLOCK_Q * BLOCK_KV * C) scalar loop with a single
// 8×8 simdgroup_multiply_accumulate call. Same memory pattern as naive
// (K/V reloaded per pass) so smem stays within 32KB.
//
//   bwd_dq_flash   : grid (cdiv(max_q_seqlen, BLOCK_Q), H, N).
//     Pass 1 — S = Q @ K^T (simdgroup matmul) → online softmax m, l.
//     Pass 2 — S redo, P = exp(S-m)/l, dP = dO @ V^T (simdgroup matmul),
//              D += rowsum(P * dP).
//     Pass 3 — S redo, P, dP redo, dS = P*(dP-D), dQ += dS @ K (simdgroup).
//     Writes dQ (ELEM_T) and aux (m, l, D) in fp32.
//
//   bwd_dkdv_flash : grid (cdiv(max_kv_seqlen, BLOCK_KV), H, N).
//     Single pass over Q tiles. Reads (m, l, D) from aux. For each Q tile:
//       S = Q @ K^T, P = exp(S-m)/l, dP = dO @ V^T, dS = P*(dP-D).
//       dV_acc += P^T @ dO, dK_acc += dS^T @ Q * scale (simdgroup matmul,
//       accumulators in registers).
//     Writes dK, dV.
//
// Math (same as naive bwd):
//   P  = softmax(Q K^T * scale)
//   dV = P^T @ dO
//   dP = dO @ V^T
//   D  = rowsum(P * dP)
//   dS = P * (dP - D)
//   dQ = dS @ K * scale
//   dK = dS^T @ Q * scale
//
// Smem (fp32, C=64):
//   bwd_dq  : Q 4K + dO 4K + K 8K + V 8K + S 2K + P 2K + m/l/D 0.2K = 28.2K
//   bwd_dkdv: K 4K + V 4K + Q 8K + dO 8K + S 2K + P 2K + m/l/D 0.4K = 28.4K
// ============================================================================

#define BLOCK_Q 16
#define BLOCK_KV 32
#define MAX_HEAD_DIM 64
#define TG_THREADS 64
#define SENTINEL_NEG_INF -1.0e30f

// ---------------------------------------------------------------------------
// bwd_dq: 3-pass per Q-tile (matches naive structure, simdgroup matmul inside)
// ---------------------------------------------------------------------------
#define SPARSE_ATTN_BWD_DQ_KERNEL(NAME, ELEM_T)                                             \
kernel void NAME(                                                                           \
    const device ELEM_T* q                [[buffer(0)]],                                    \
    const device ELEM_T* k                [[buffer(1)]],                                    \
    const device ELEM_T* v                [[buffer(2)]],                                    \
    const device ELEM_T* d_out            [[buffer(3)]],                                    \
    const device int*    cu_seqlens_q     [[buffer(4)]],                                    \
    const device int*    cu_seqlens_kv    [[buffer(5)]],                                    \
    device ELEM_T*       d_q              [[buffer(6)]],                                    \
    device float*        m_aux            [[buffer(7)]],                                    \
    device float*        l_aux            [[buffer(8)]],                                    \
    device float*        d_aux            [[buffer(9)]],                                    \
    constant uint&       H                [[buffer(10)]],                                   \
    constant uint&       C_q              [[buffer(11)]],                                   \
    constant uint&       C_v              [[buffer(12)]],                                   \
    constant float&      scale            [[buffer(13)]],                                   \
    threadgroup uchar*   smem_raw         [[threadgroup(0)]],                               \
    uint3 gid                             [[threadgroup_position_in_grid]],                 \
    uint  lid                             [[thread_index_in_threadgroup]],                  \
    uint  simd_id                         [[simdgroup_index_in_threadgroup]]                \
) {                                                                                         \
    uint q_tile_idx = gid.x;                                                                \
    uint h          = gid.y;                                                                \
    uint seq        = gid.z;                                                                \
    if (h >= H) return;                                                                     \
                                                                                            \
    uint q_start = (uint)cu_seqlens_q[seq];                                                 \
    uint q_end   = (uint)cu_seqlens_q[seq + 1];                                             \
    uint q_len   = q_end - q_start;                                                         \
    uint q_tile_start = q_tile_idx * BLOCK_Q;                                               \
    if (q_tile_start >= q_len) return;                                                      \
    uint q_tile_rows = min((uint)BLOCK_Q, q_len - q_tile_start);                            \
                                                                                            \
    uint kv_start = (uint)cu_seqlens_kv[seq];                                               \
    uint kv_end   = (uint)cu_seqlens_kv[seq + 1];                                           \
    uint kv_len   = kv_end - kv_start;                                                      \
                                                                                            \
    threadgroup ELEM_T* smem_q   = (threadgroup ELEM_T*)smem_raw;                           \
    threadgroup ELEM_T* smem_dO  = smem_q + BLOCK_Q  * C_q;                                 \
    threadgroup ELEM_T* smem_k   = smem_dO + BLOCK_Q * C_v;                                 \
    threadgroup ELEM_T* smem_v   = smem_k  + BLOCK_KV * C_q;                                \
    threadgroup float*  smem_s   = (threadgroup float*)(smem_v + BLOCK_KV * C_v);           \
    threadgroup ELEM_T* smem_p   = (threadgroup ELEM_T*)(smem_s + BLOCK_Q * BLOCK_KV);      \
    threadgroup float*  smem_m   = (threadgroup float*)(smem_p + BLOCK_Q * BLOCK_KV);       \
    threadgroup float*  smem_l   = smem_m + BLOCK_Q;                                        \
    threadgroup float*  smem_D   = smem_l + BLOCK_Q;                                        \
                                                                                            \
    /* Load Q, dO. */                                                                       \
    for (uint i = lid; i < BLOCK_Q * C_q; i += TG_THREADS) {                                \
        uint row = i / C_q, col = i % C_q;                                                  \
        bool valid = row < q_tile_rows;                                                     \
        uint q_row_global = q_start + q_tile_start + row;                                   \
        smem_q[row * C_q + col] = valid                                                     \
            ? q[(q_row_global * H + h) * C_q + col] : (ELEM_T)0;                            \
    }                                                                                       \
    for (uint i = lid; i < BLOCK_Q * C_v; i += TG_THREADS) {                                \
        uint row = i / C_v, col = i % C_v;                                                  \
        bool valid = row < q_tile_rows;                                                     \
        uint q_row_global = q_start + q_tile_start + row;                                   \
        smem_dO[row * C_v + col] = valid                                                    \
            ? d_out[(q_row_global * H + h) * C_v + col] : (ELEM_T)0;                        \
    }                                                                                       \
    if (lid < BLOCK_Q) {                                                                    \
        smem_m[lid] = SENTINEL_NEG_INF;                                                     \
        smem_l[lid] = 0.0f;                                                                 \
        smem_D[lid] = 0.0f;                                                                 \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
                                                                                            \
    /* ======================================================================== */         \
    /* Pass 1: compute (m, l) per Q row.                                        */         \
    /* ======================================================================== */         \
    for (uint kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += BLOCK_KV) {       \
        uint kv_tile_rows = min((uint)BLOCK_KV, kv_len - kv_tile_start);                    \
        for (uint i = lid; i < BLOCK_KV * C_q; i += TG_THREADS) {                           \
            uint row = i / C_q, col = i % C_q;                                              \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_k[row * C_q + col] = valid                                                 \
                ? k[(kv_row_global * H + h) * C_q + col] : (ELEM_T)0;                       \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            simdgroup_matrix<float, 8, 8> s_acc[BLOCK_KV / 8];                              \
            for (uint t = 0; t < BLOCK_KV / 8; t++)                                         \
                s_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                               \
            for (uint k8 = 0; k8 < C_q; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_q + simd_id * 8 * C_q + k8, C_q);                \
                for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_k + tn * 8 * C_q + k8, C_q,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(s_acc[tn], a_mat, b_mat, s_acc[tn]);      \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                    \
                simdgroup_store(s_acc[tn], smem_s + simd_id * 8 * BLOCK_KV + tn * 8,        \
                                BLOCK_KV);                                                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* Online softmax update, 4 threads per row. */                                     \
        {                                                                                   \
            uint row       = lid / 4;                                                       \
            uint col_group = lid % 4;                                                       \
            uint s_col_start = col_group * (BLOCK_KV / 4);                                  \
            uint s_col_end   = s_col_start + (BLOCK_KV / 4);                                \
            float partial_max = SENTINEL_NEG_INF;                                           \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float s_val = (j < kv_tile_rows)                                            \
                    ? smem_s[row * BLOCK_KV + j] * scale : SENTINEL_NEG_INF;                \
                if (s_val > partial_max) partial_max = s_val;                               \
            }                                                                               \
            float row_max = partial_max;                                                    \
            row_max = max(row_max, simd_shuffle_xor(row_max, 1u));                          \
            row_max = max(row_max, simd_shuffle_xor(row_max, 2u));                          \
            float m_old = smem_m[row];                                                      \
            float m_new = max(m_old, row_max);                                              \
            float alpha = exp(m_old - m_new);                                               \
            float partial_sum = 0.0f;                                                       \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float s_val = (j < kv_tile_rows)                                            \
                    ? smem_s[row * BLOCK_KV + j] * scale : SENTINEL_NEG_INF;                \
                partial_sum += exp(s_val - m_new);                                          \
            }                                                                               \
            float row_sum = partial_sum;                                                    \
            row_sum += simd_shuffle_xor(row_sum, 1u);                                       \
            row_sum += simd_shuffle_xor(row_sum, 2u);                                       \
            if (col_group == 0 && row < q_tile_rows) {                                      \
                smem_l[row] = smem_l[row] * alpha + row_sum;                                \
                smem_m[row] = m_new;                                                        \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    }                                                                                       \
                                                                                            \
    /* ======================================================================== */         \
    /* Pass 2: D += rowsum(P * dP)                                              */         \
    /* ======================================================================== */         \
    for (uint kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += BLOCK_KV) {       \
        uint kv_tile_rows = min((uint)BLOCK_KV, kv_len - kv_tile_start);                    \
        for (uint i = lid; i < BLOCK_KV * C_q; i += TG_THREADS) {                           \
            uint row = i / C_q, col = i % C_q;                                              \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_k[row * C_q + col] = valid                                                 \
                ? k[(kv_row_global * H + h) * C_q + col] : (ELEM_T)0;                       \
        }                                                                                   \
        for (uint i = lid; i < BLOCK_KV * C_v; i += TG_THREADS) {                           \
            uint row = i / C_v, col = i % C_v;                                              \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_v[row * C_v + col] = valid                                                 \
                ? v[(kv_row_global * H + h) * C_v + col] : (ELEM_T)0;                       \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* S = Q @ K^T. */                                                                  \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            simdgroup_matrix<float, 8, 8> s_acc[BLOCK_KV / 8];                              \
            for (uint t = 0; t < BLOCK_KV / 8; t++)                                         \
                s_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                               \
            for (uint k8 = 0; k8 < C_q; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_q + simd_id * 8 * C_q + k8, C_q);                \
                for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_k + tn * 8 * C_q + k8, C_q,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(s_acc[tn], a_mat, b_mat, s_acc[tn]);      \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                    \
                simdgroup_store(s_acc[tn], smem_s + simd_id * 8 * BLOCK_KV + tn * 8,        \
                                BLOCK_KV);                                                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* Compute P = exp(S*scale - m) / l into smem_p (ELEM_T). */                        \
        {                                                                                   \
            uint row       = lid / 4;                                                       \
            uint col_group = lid % 4;                                                       \
            uint s_col_start = col_group * (BLOCK_KV / 4);                                  \
            uint s_col_end   = s_col_start + (BLOCK_KV / 4);                                \
            float m_row = smem_m[row];                                                      \
            float l_row = smem_l[row];                                                      \
            float inv_l = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;                           \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float s_val = smem_s[row * BLOCK_KV + j] * scale;                           \
                float p_val = (j < kv_tile_rows) ? exp(s_val - m_row) * inv_l : 0.0f;       \
                smem_p[row * BLOCK_KV + j] = (ELEM_T)p_val;                                 \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dP = dO @ V^T into smem_s (overwriting S). */                                    \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            simdgroup_matrix<float, 8, 8> dp_acc[BLOCK_KV / 8];                             \
            for (uint t = 0; t < BLOCK_KV / 8; t++)                                         \
                dp_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                              \
            for (uint k8 = 0; k8 < C_v; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_dO + simd_id * 8 * C_v + k8, C_v);               \
                for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_v + tn * 8 * C_v + k8, C_v,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(dp_acc[tn], a_mat, b_mat, dp_acc[tn]);    \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                    \
                simdgroup_store(dp_acc[tn], smem_s + simd_id * 8 * BLOCK_KV + tn * 8,       \
                                BLOCK_KV);                                                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* D += rowsum(P * dP). */                                                          \
        {                                                                                   \
            uint row       = lid / 4;                                                       \
            uint col_group = lid % 4;                                                       \
            uint s_col_start = col_group * (BLOCK_KV / 4);                                  \
            uint s_col_end   = s_col_start + (BLOCK_KV / 4);                                \
            float partial_d = 0.0f;                                                         \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float p_val  = (float)smem_p[row * BLOCK_KV + j];                           \
                float dp_val = smem_s[row * BLOCK_KV + j];                                  \
                if (j < kv_tile_rows) partial_d += p_val * dp_val;                          \
            }                                                                               \
            float d_add = partial_d;                                                        \
            d_add += simd_shuffle_xor(d_add, 1u);                                           \
            d_add += simd_shuffle_xor(d_add, 2u);                                           \
            if (col_group == 0 && row < q_tile_rows) {                                      \
                smem_D[row] += d_add;                                                       \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    }                                                                                       \
                                                                                            \
    /* ======================================================================== */         \
    /* Pass 3: dQ = sum_j (P * (dP - D))_j * K_j * scale via simdgroup matmul.  */         \
    /* dQ accumulator held in registers (C_q/8 tiles per simdgroup).            */         \
    /* ======================================================================== */         \
    simdgroup_matrix<float, 8, 8> dq_acc[MAX_HEAD_DIM / 8];                                 \
    for (uint tn = 0; tn < MAX_HEAD_DIM / 8; tn++)                                          \
        dq_acc[tn] = simdgroup_matrix<float, 8, 8>(0.0f);                                   \
                                                                                            \
    for (uint kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += BLOCK_KV) {       \
        uint kv_tile_rows = min((uint)BLOCK_KV, kv_len - kv_tile_start);                    \
        for (uint i = lid; i < BLOCK_KV * C_q; i += TG_THREADS) {                           \
            uint row = i / C_q, col = i % C_q;                                              \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_k[row * C_q + col] = valid                                                 \
                ? k[(kv_row_global * H + h) * C_q + col] : (ELEM_T)0;                       \
        }                                                                                   \
        for (uint i = lid; i < BLOCK_KV * C_v; i += TG_THREADS) {                           \
            uint row = i / C_v, col = i % C_v;                                              \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_v[row * C_v + col] = valid                                                 \
                ? v[(kv_row_global * H + h) * C_v + col] : (ELEM_T)0;                       \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* S = Q @ K^T again. */                                                            \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            simdgroup_matrix<float, 8, 8> s_acc[BLOCK_KV / 8];                              \
            for (uint t = 0; t < BLOCK_KV / 8; t++)                                         \
                s_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                               \
            for (uint k8 = 0; k8 < C_q; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_q + simd_id * 8 * C_q + k8, C_q);                \
                for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_k + tn * 8 * C_q + k8, C_q,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(s_acc[tn], a_mat, b_mat, s_acc[tn]);      \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                    \
                simdgroup_store(s_acc[tn], smem_s + simd_id * 8 * BLOCK_KV + tn * 8,        \
                                BLOCK_KV);                                                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* Compute P = exp(S*scale - m) / l → smem_p. */                                    \
        {                                                                                   \
            uint row       = lid / 4;                                                       \
            uint col_group = lid % 4;                                                       \
            uint s_col_start = col_group * (BLOCK_KV / 4);                                  \
            uint s_col_end   = s_col_start + (BLOCK_KV / 4);                                \
            float m_row = smem_m[row];                                                      \
            float l_row = smem_l[row];                                                      \
            float inv_l = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;                           \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float s_val = smem_s[row * BLOCK_KV + j] * scale;                           \
                float p_val = (j < kv_tile_rows) ? exp(s_val - m_row) * inv_l : 0.0f;       \
                smem_p[row * BLOCK_KV + j] = (ELEM_T)p_val;                                 \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dP = dO @ V^T into smem_s. */                                                    \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            simdgroup_matrix<float, 8, 8> dp_acc[BLOCK_KV / 8];                             \
            for (uint t = 0; t < BLOCK_KV / 8; t++)                                         \
                dp_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                              \
            for (uint k8 = 0; k8 < C_v; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_dO + simd_id * 8 * C_v + k8, C_v);               \
                for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_v + tn * 8 * C_v + k8, C_v,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(dp_acc[tn], a_mat, b_mat, dp_acc[tn]);    \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                    \
                simdgroup_store(dp_acc[tn], smem_s + simd_id * 8 * BLOCK_KV + tn * 8,       \
                                BLOCK_KV);                                                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dS = P * (dP - D). Write into smem_p (ELEM_T) for the dQ @ K matmul. */          \
        {                                                                                   \
            uint row       = lid / 4;                                                       \
            uint col_group = lid % 4;                                                       \
            uint s_col_start = col_group * (BLOCK_KV / 4);                                  \
            uint s_col_end   = s_col_start + (BLOCK_KV / 4);                                \
            float D_row = smem_D[row];                                                      \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float p_val  = (float)smem_p[row * BLOCK_KV + j];                           \
                float dp_val = smem_s[row * BLOCK_KV + j];                                  \
                float ds_val = (j < kv_tile_rows)                                           \
                    ? p_val * (dp_val - D_row) : 0.0f;                                      \
                smem_p[row * BLOCK_KV + j] = (ELEM_T)ds_val;                                \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dQ_acc += dS @ K via simdgroup matmul. smem_p holds dS (ELEM_T),   */            \
        /*                                        smem_k holds K (ELEM_T).   */             \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            uint n_tiles_cq = C_q / 8;                                                      \
            for (uint k8 = 0; k8 < BLOCK_KV; k8 += 8) {                                     \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_p + simd_id * 8 * BLOCK_KV + k8, BLOCK_KV);      \
                for (uint tn = 0; tn < n_tiles_cq; tn++) {                                  \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_k + k8 * C_q + tn * 8, C_q);                 \
                    simdgroup_multiply_accumulate(dq_acc[tn], a_mat, b_mat, dq_acc[tn]);    \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    }                                                                                       \
                                                                                            \
    /* Write aux (m, l, D) for bwd_dkdv BEFORE the dQ spill — the spill scratch    */      \
    /* (smem_s, BLOCK_Q*C_q floats = 4KB) overflows smem_p at fp16/bf16 (where    */       \
    /* smem_s + smem_p combined is only 3KB) and into smem_m/l/D, clobbering them  */      \
    /* before they can be flushed to global. At fp32, smem_s + smem_p = 4KB so the */      \
    /* overflow landed exactly at the end of smem_p (harmless since it wasn't read */      \
    /* again). This is exactly the mechanism that broke bwd_dkdv fp16 C=64.        */      \
    if (lid < BLOCK_Q && lid < q_tile_rows) {                                               \
        uint q_row_global = q_start + q_tile_start + lid;                                   \
        m_aux[q_row_global * H + h] = smem_m[lid];                                          \
        l_aux[q_row_global * H + h] = smem_l[lid];                                          \
        d_aux[q_row_global * H + h] = smem_D[lid];                                          \
    }                                                                                       \
                                                                                            \
    /* Spill dQ_acc to smem_s+smem_p contiguous (4K fp32 = BLOCK_Q * C_q floats)  */        \
    /* and write to global with * scale. dS in smem_p is no longer needed here.   */        \
    {                                                                                       \
        threadgroup float* dq_scratch = smem_s;                                             \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            uint n_tiles_cq = C_q / 8;                                                      \
            for (uint tn = 0; tn < n_tiles_cq; tn++) {                                      \
                simdgroup_store(dq_acc[tn], dq_scratch + simd_id * 8 * C_q + tn * 8, C_q);  \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    {                                                                                       \
        threadgroup float* dq_scratch = smem_s;                                             \
        for (uint i = lid; i < BLOCK_Q * C_q; i += TG_THREADS) {                            \
            uint row = i / C_q, col = i % C_q;                                              \
            if (row < q_tile_rows) {                                                        \
                uint q_row_global = q_start + q_tile_start + row;                           \
                d_q[(q_row_global * H + h) * C_q + col] =                                   \
                    (ELEM_T)(dq_scratch[row * C_q + col] * scale);                          \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
}

SPARSE_ATTN_BWD_DQ_KERNEL(sparse_attention_bwd_dq_flash,        float)
SPARSE_ATTN_BWD_DQ_KERNEL(sparse_attention_bwd_dq_flash_half,   half)
SPARSE_ATTN_BWD_DQ_KERNEL(sparse_attention_bwd_dq_flash_bfloat, bfloat)


// ---------------------------------------------------------------------------
// bwd_dkdv: single Q-tile pass per KV-tile with simdgroup matmul.
// ---------------------------------------------------------------------------
//
// NOTE: this kernel uses BLOCK_KV_K=16 (inner KV-tile size for accumulator
// rows) and BLOCK_Q_K=32 (inner Q-tile size). This is independent of the
// bwd_dq BLOCK_Q/BLOCK_KV — different symmetry.
// ---------------------------------------------------------------------------
#define BLOCK_KV_K 16
#define BLOCK_Q_K  32

#define SPARSE_ATTN_BWD_DKDV_KERNEL(NAME, ELEM_T)                                           \
kernel void NAME(                                                                           \
    const device ELEM_T* q                [[buffer(0)]],                                    \
    const device ELEM_T* k                [[buffer(1)]],                                    \
    const device ELEM_T* v                [[buffer(2)]],                                    \
    const device ELEM_T* d_out            [[buffer(3)]],                                    \
    const device int*    cu_seqlens_q     [[buffer(4)]],                                    \
    const device int*    cu_seqlens_kv    [[buffer(5)]],                                    \
    const device float*  m_aux            [[buffer(6)]],                                    \
    const device float*  l_aux            [[buffer(7)]],                                    \
    const device float*  d_aux            [[buffer(8)]],                                    \
    device ELEM_T*       d_k              [[buffer(9)]],                                    \
    device ELEM_T*       d_v              [[buffer(10)]],                                   \
    constant uint&       H                [[buffer(11)]],                                   \
    constant uint&       C_q              [[buffer(12)]],                                   \
    constant uint&       C_v              [[buffer(13)]],                                   \
    constant float&      scale            [[buffer(14)]],                                   \
    threadgroup uchar*   smem_raw         [[threadgroup(0)]],                               \
    uint3 gid                             [[threadgroup_position_in_grid]],                 \
    uint  lid                             [[thread_index_in_threadgroup]],                  \
    uint  simd_id                         [[simdgroup_index_in_threadgroup]]                \
) {                                                                                         \
    uint kv_tile_idx = gid.x;                                                               \
    uint h           = gid.y;                                                               \
    uint seq         = gid.z;                                                               \
    if (h >= H) return;                                                                     \
                                                                                            \
    uint kv_start = (uint)cu_seqlens_kv[seq];                                               \
    uint kv_end   = (uint)cu_seqlens_kv[seq + 1];                                           \
    uint kv_len   = kv_end - kv_start;                                                      \
    uint kv_tile_start = kv_tile_idx * BLOCK_KV_K;                                          \
    if (kv_tile_start >= kv_len) return;                                                    \
    uint kv_tile_rows = min((uint)BLOCK_KV_K, kv_len - kv_tile_start);                      \
                                                                                            \
    uint q_start = (uint)cu_seqlens_q[seq];                                                 \
    uint q_end   = (uint)cu_seqlens_q[seq + 1];                                             \
    uint q_len   = q_end - q_start;                                                         \
                                                                                            \
    threadgroup ELEM_T* smem_k   = (threadgroup ELEM_T*)smem_raw;                           \
    threadgroup ELEM_T* smem_v   = smem_k + BLOCK_KV_K * C_q;                               \
    threadgroup ELEM_T* smem_q   = smem_v + BLOCK_KV_K * C_v;                               \
    threadgroup ELEM_T* smem_dO  = smem_q + BLOCK_Q_K * C_q;                                \
    threadgroup float*  smem_s   = (threadgroup float*)(smem_dO + BLOCK_Q_K * C_v);         \
    threadgroup ELEM_T* smem_p   = (threadgroup ELEM_T*)(smem_s + BLOCK_Q_K * BLOCK_KV_K);  \
    /* smem_pT — [BLOCK_KV_K, BLOCK_Q_K] ELEM_T. Holds P^T during dV matmul,     */        \
    /* then dS^T during dK matmul. Lets both matmuls use non-transpose loads —   */        \
    /* the fp16 transpose-load on this small-stride buffer was buggy in          */        \
    /* Metal for our Apple Silicon toolchain (produced 1000× wrong values        */        \
    /* despite fp16 transpose-load on K with large stride working fine).         */        \
    threadgroup ELEM_T* smem_pT  = smem_p  + BLOCK_Q_K * BLOCK_KV_K;                        \
    threadgroup float*  smem_m   = (threadgroup float*)(smem_pT + BLOCK_KV_K * BLOCK_Q_K);  \
    threadgroup float*  smem_l   = smem_m + BLOCK_Q_K;                                      \
    threadgroup float*  smem_D   = smem_l + BLOCK_Q_K;                                      \
                                                                                            \
    /* Load K, V tile (once). */                                                            \
    for (uint i = lid; i < BLOCK_KV_K * C_q; i += TG_THREADS) {                             \
        uint row = i / C_q, col = i % C_q;                                                  \
        bool valid = row < kv_tile_rows;                                                    \
        uint kv_row_global = kv_start + kv_tile_start + row;                                \
        smem_k[row * C_q + col] = valid                                                     \
            ? k[(kv_row_global * H + h) * C_q + col] : (ELEM_T)0;                           \
    }                                                                                       \
    for (uint i = lid; i < BLOCK_KV_K * C_v; i += TG_THREADS) {                             \
        uint row = i / C_v, col = i % C_v;                                                  \
        bool valid = row < kv_tile_rows;                                                    \
        uint kv_row_global = kv_start + kv_tile_start + row;                                \
        smem_v[row * C_v + col] = valid                                                     \
            ? v[(kv_row_global * H + h) * C_v + col] : (ELEM_T)0;                           \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
                                                                                            \
    /* Accumulators in registers: 2 simdgroups, each owns 8 KV rows × C_q (dK) or C_v (dV). */\
    simdgroup_matrix<float, 8, 8> dk_acc[MAX_HEAD_DIM / 8];                                 \
    simdgroup_matrix<float, 8, 8> dv_acc[MAX_HEAD_DIM / 8];                                 \
    for (uint tn = 0; tn < MAX_HEAD_DIM / 8; tn++) {                                        \
        dk_acc[tn] = simdgroup_matrix<float, 8, 8>(0.0f);                                   \
        dv_acc[tn] = simdgroup_matrix<float, 8, 8>(0.0f);                                   \
    }                                                                                       \
                                                                                            \
    /* Loop over Q tiles. */                                                                \
    for (uint q_tile_start = 0; q_tile_start < q_len; q_tile_start += BLOCK_Q_K) {          \
        uint q_tile_rows = min((uint)BLOCK_Q_K, q_len - q_tile_start);                      \
        for (uint i = lid; i < BLOCK_Q_K * C_q; i += TG_THREADS) {                          \
            uint row = i / C_q, col = i % C_q;                                              \
            bool valid = row < q_tile_rows;                                                 \
            uint q_row_global = q_start + q_tile_start + row;                               \
            smem_q[row * C_q + col] = valid                                                 \
                ? q[(q_row_global * H + h) * C_q + col] : (ELEM_T)0;                        \
        }                                                                                   \
        for (uint i = lid; i < BLOCK_Q_K * C_v; i += TG_THREADS) {                          \
            uint row = i / C_v, col = i % C_v;                                              \
            bool valid = row < q_tile_rows;                                                 \
            uint q_row_global = q_start + q_tile_start + row;                               \
            smem_dO[row * C_v + col] = valid                                                \
                ? d_out[(q_row_global * H + h) * C_v + col] : (ELEM_T)0;                    \
        }                                                                                   \
        if (lid < BLOCK_Q_K) {                                                              \
            if (lid < q_tile_rows) {                                                        \
                uint q_row_global = q_start + q_tile_start + lid;                           \
                smem_m[lid] = m_aux[q_row_global * H + h];                                  \
                smem_l[lid] = l_aux[q_row_global * H + h];                                  \
                smem_D[lid] = d_aux[q_row_global * H + h];                                  \
            } else {                                                                        \
                smem_m[lid] = 0.0f;                                                         \
                smem_l[lid] = 1.0f;  /* safe divisor */                                     \
                smem_D[lid] = 0.0f;                                                         \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* S [Q_tile, KV_tile] = Q @ K^T. Simdgroup i owns Q rows [i*8, i*8+8).  */         \
        /* For Q_tile=32 and 2 simdgroups → each simdgroup handles 16 Q rows → 2 */         \
        /* outer 8-row strips per simdgroup. Inner: KV_tile=16 → 2 col-tiles.    */         \
        /* To keep it simple we loop the outer strip per simdgroup.              */         \
        for (uint row_strip = 0; row_strip < BLOCK_Q_K / 8; row_strip += (BLOCK_Q_K / 8 / (TG_THREADS/32))) { \
            uint row_base = (simd_id + row_strip) * 8;                                      \
            if (row_base >= BLOCK_Q_K) break;                                               \
            simdgroup_matrix<float, 8, 8> s_acc[BLOCK_KV_K / 8];                            \
            for (uint t = 0; t < BLOCK_KV_K / 8; t++)                                       \
                s_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                               \
            for (uint k8 = 0; k8 < C_q; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_q + row_base * C_q + k8, C_q);                   \
                for (uint tn = 0; tn < BLOCK_KV_K / 8; tn++) {                              \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_k + tn * 8 * C_q + k8, C_q,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(s_acc[tn], a_mat, b_mat, s_acc[tn]);      \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV_K / 8; tn++) {                                  \
                simdgroup_store(s_acc[tn], smem_s + row_base * BLOCK_KV_K + tn * 8,         \
                                BLOCK_KV_K);                                                \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* P = exp(S*scale - m) / l into smem_p (row-major [Q,KV] ELEM_T) and   */          \
        /* simultaneously into smem_pT (row-major [KV,Q] ELEM_T) so the dV      */          \
        /* matmul can read P^T without a transpose-load.                        */          \
        for (uint idx = lid; idx < BLOCK_Q_K * BLOCK_KV_K; idx += TG_THREADS) {             \
            uint row = idx / BLOCK_KV_K, col = idx % BLOCK_KV_K;                            \
            float s_val = smem_s[row * BLOCK_KV_K + col] * scale;                           \
            float m_row = smem_m[row];                                                      \
            float l_row = smem_l[row];                                                      \
            float inv_l = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;                           \
            bool valid = (row < q_tile_rows) && (col < kv_tile_rows);                       \
            float p_val = valid ? exp(s_val - m_row) * inv_l : 0.0f;                        \
            smem_p [row * BLOCK_KV_K + col] = (ELEM_T)p_val;                                \
            smem_pT[col * BLOCK_Q_K  + row] = (ELEM_T)p_val;                                \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dP = dO @ V^T into smem_s (overwriting S). */                                    \
        for (uint row_strip = 0; row_strip < BLOCK_Q_K / 8; row_strip += (BLOCK_Q_K / 8 / (TG_THREADS/32))) { \
            uint row_base = (simd_id + row_strip) * 8;                                      \
            if (row_base >= BLOCK_Q_K) break;                                               \
            simdgroup_matrix<float, 8, 8> dp_acc[BLOCK_KV_K / 8];                           \
            for (uint t = 0; t < BLOCK_KV_K / 8; t++)                                       \
                dp_acc[t] = simdgroup_matrix<float,8,8>(0.0f);                              \
            for (uint k8 = 0; k8 < C_v; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_dO + row_base * C_v + k8, C_v);                  \
                for (uint tn = 0; tn < BLOCK_KV_K / 8; tn++) {                              \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_v + tn * 8 * C_v + k8, C_v,                  \
                                   ulong2(0, 0), true);                                     \
                    simdgroup_multiply_accumulate(dp_acc[tn], a_mat, b_mat, dp_acc[tn]);    \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV_K / 8; tn++) {                                  \
                simdgroup_store(dp_acc[tn], smem_s + row_base * BLOCK_KV_K + tn * 8,        \
                                BLOCK_KV_K);                                                \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dV_acc += P^T @ dO. Each simdgroup owns 8 KV rows (output rows of dV). */        \
        /* A = P^T[8, 8] loaded from smem_pT with plain (non-transpose) load.    */         \
        if (simd_id < (BLOCK_KV_K / 8)) {                                                   \
            uint n_tiles_cv = C_v / 8;                                                      \
            for (uint k8 = 0; k8 < BLOCK_Q_K; k8 += 8) {                                    \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_pT + simd_id * 8 * BLOCK_Q_K + k8, BLOCK_Q_K);   \
                for (uint tn = 0; tn < n_tiles_cv; tn++) {                                  \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_dO + k8 * C_v + tn * 8, C_v);                \
                    simdgroup_multiply_accumulate(dv_acc[tn], a_mat, b_mat, dv_acc[tn]);    \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dS = P * (dP - D). Write to both smem_p (row-major [Q,KV]) and          */       \
        /* smem_pT (row-major [KV,Q] — i.e. dS^T) so the dK matmul can use a       */       \
        /* plain non-transpose load. smem_p holds dS in case any future change     */       \
        /* wants to read it without transpose.                                     */       \
        for (uint idx = lid; idx < BLOCK_Q_K * BLOCK_KV_K; idx += TG_THREADS) {             \
            uint row = idx / BLOCK_KV_K, col = idx % BLOCK_KV_K;                            \
            float p_val  = (float)smem_p[row * BLOCK_KV_K + col];                           \
            float dp_val = smem_s[row * BLOCK_KV_K + col];                                  \
            float D_row  = smem_D[row];                                                     \
            bool valid = (row < q_tile_rows) && (col < kv_tile_rows);                       \
            float ds_val = valid ? p_val * (dp_val - D_row) : 0.0f;                         \
            smem_p [row * BLOCK_KV_K + col] = (ELEM_T)ds_val;                               \
            smem_pT[col * BLOCK_Q_K  + row] = (ELEM_T)ds_val;                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* dK_acc += dS^T @ Q. A loaded from smem_pT (non-transpose) B from smem_q. */      \
        if (simd_id < (BLOCK_KV_K / 8)) {                                                   \
            uint n_tiles_cq = C_q / 8;                                                      \
            for (uint k8 = 0; k8 < BLOCK_Q_K; k8 += 8) {                                    \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_pT + simd_id * 8 * BLOCK_Q_K + k8, BLOCK_Q_K);   \
                for (uint tn = 0; tn < n_tiles_cq; tn++) {                                  \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_q + k8 * C_q + tn * 8, C_q);                 \
                    simdgroup_multiply_accumulate(dk_acc[tn], a_mat, b_mat, dk_acc[tn]);    \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    }                                                                                       \
                                                                                            \
    /* Write dV then dK. Spill via combined smem_s+smem_p scratch region (4K,    */         \
    /* = BLOCK_KV_K * C_v fp32 = BLOCK_KV_K * C_q fp32). Stride of each          */         \
    /* simdgroup_store is the target's real leading dimension (C_v/C_q), not    */         \
    /* BLOCK_KV_K — those two differ once C > BLOCK_KV_K.                       */         \
    {                                                                                       \
        threadgroup float* scratch = smem_s;                                                \
        if (simd_id < (BLOCK_KV_K / 8)) {                                                   \
            uint n_tiles_cv = C_v / 8;                                                      \
            for (uint tn = 0; tn < n_tiles_cv; tn++) {                                      \
                simdgroup_store(dv_acc[tn], scratch + simd_id * 8 * C_v + tn * 8, C_v);     \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    {                                                                                       \
        threadgroup float* scratch = smem_s;                                                \
        for (uint i = lid; i < BLOCK_KV_K * C_v; i += TG_THREADS) {                         \
            uint row = i / C_v, col = i % C_v;                                              \
            if (row < kv_tile_rows) {                                                       \
                uint kv_row_global = kv_start + kv_tile_start + row;                        \
                d_v[(kv_row_global * H + h) * C_v + col] =                                  \
                    (ELEM_T)scratch[row * C_v + col];                                       \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    {                                                                                       \
        threadgroup float* scratch = smem_s;                                                \
        if (simd_id < (BLOCK_KV_K / 8)) {                                                   \
            uint n_tiles_cq = C_q / 8;                                                      \
            for (uint tn = 0; tn < n_tiles_cq; tn++) {                                      \
                simdgroup_store(dk_acc[tn], scratch + simd_id * 8 * C_q + tn * 8, C_q);     \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    {                                                                                       \
        threadgroup float* scratch = smem_s;                                                \
        for (uint i = lid; i < BLOCK_KV_K * C_q; i += TG_THREADS) {                         \
            uint row = i / C_q, col = i % C_q;                                              \
            if (row < kv_tile_rows) {                                                       \
                uint kv_row_global = kv_start + kv_tile_start + row;                        \
                d_k[(kv_row_global * H + h) * C_q + col] =                                  \
                    (ELEM_T)(scratch[row * C_q + col] * scale);                             \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
}

SPARSE_ATTN_BWD_DKDV_KERNEL(sparse_attention_bwd_dkdv_flash,        float)
SPARSE_ATTN_BWD_DKDV_KERNEL(sparse_attention_bwd_dkdv_flash_half,   half)
SPARSE_ATTN_BWD_DKDV_KERNEL(sparse_attention_bwd_dkdv_flash_bfloat, bfloat)
