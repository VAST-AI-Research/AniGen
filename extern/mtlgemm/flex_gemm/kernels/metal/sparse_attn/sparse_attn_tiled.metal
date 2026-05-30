#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Flash-attention-v2 forward for variable-length sparse sequences.
//
// Uses simdgroup_matrix_multiply_accumulate for Q @ K^T and P @ V, matching
// the SPCONV_FWD_KERNEL pattern in flex_gemm/kernels/metal/spconv/spconv_gemm.metal.
//
// Design:
//   BLOCK_Q = 16, BLOCK_KV = 32, HEAD_DIM <= 64 (must be multiple of 8).
//   Threadgroup = 64 threads = 2 simdgroups. Simdgroup i owns Q rows [i*8, i*8+8).
//   Grid: (cdiv(max_q_seqlen, BLOCK_Q), H, N).
//
// Algorithm (flash-attention-v2 per threadgroup, online softmax):
//   1. Load Q-tile [BLOCK_Q, C_q] into smem once.
//   2. For each KV-tile of BLOCK_KV rows:
//        a. Load K, V into smem cooperatively.
//        b. S = Q @ K^T via simdgroup matmul (8-row strip per simdgroup).
//        c. Per-row online softmax: row-max, rescale O, compute P = exp(S-m).
//        d. O += P @ V via simdgroup matmul.
//   3. Normalize O /= l and write to global.
//
// Smem layout (dynamically sized by dispatcher based on dtype and C_q, C_v):
//   smem_q [BLOCK_Q, C_q]     ELEM_T
//   smem_k [BLOCK_KV, C_q]    ELEM_T
//   smem_v [BLOCK_KV, C_v]    ELEM_T
//   smem_s [BLOCK_Q, BLOCK_KV] float
//   smem_p [BLOCK_Q, BLOCK_KV] ELEM_T  (cast of S, matches V dtype for P@V)
//   smem_o [BLOCK_Q, C_v]     float
//   smem_m [BLOCK_Q]          float
//   smem_l [BLOCK_Q]          float
//
// For fp16 + head_dim=64: ~17KB. For fp32 + head_dim=64: ~28KB. Both fit
// within 32KB threadgroup memory on Apple Silicon.
// ============================================================================

#define BLOCK_Q 16
#define BLOCK_KV 32
#define MAX_HEAD_DIM 64
#define TG_THREADS 64
#define SENTINEL_NEG_INF -1.0e30f

#define SPARSE_ATTN_FLASH_FWD_KERNEL(NAME, ELEM_T)                                          \
kernel void NAME(                                                                           \
    const device ELEM_T* q                [[buffer(0)]],  /* [T_q, H, C_q] */              \
    const device ELEM_T* k                [[buffer(1)]],  /* [T_kv, H, C_q] */             \
    const device ELEM_T* v                [[buffer(2)]],  /* [T_kv, H, C_v] */             \
    const device int*    cu_seqlens_q     [[buffer(3)]],  /* [N+1] */                      \
    const device int*    cu_seqlens_kv    [[buffer(4)]],  /* [N+1] */                      \
    device ELEM_T*       out              [[buffer(5)]],  /* [T_q, H, C_v] */              \
    constant uint&       H                [[buffer(6)]],                                    \
    constant uint&       C_q              [[buffer(7)]],                                    \
    constant uint&       C_v              [[buffer(8)]],                                    \
    constant float&      scale            [[buffer(9)]],                                    \
    threadgroup uchar*   smem_raw         [[threadgroup(0)]],                               \
    uint3 gid                             [[threadgroup_position_in_grid]],                 \
    uint  lid                             [[thread_index_in_threadgroup]],                  \
    uint  simd_id                         [[simdgroup_index_in_threadgroup]]                \
) {                                                                                         \
    uint q_tile_idx = gid.x;                                                                \
    uint h         = gid.y;                                                                 \
    uint seq       = gid.z;                                                                 \
                                                                                            \
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
    threadgroup ELEM_T* smem_q = (threadgroup ELEM_T*)smem_raw;                             \
    threadgroup ELEM_T* smem_k = smem_q + BLOCK_Q  * C_q;                                   \
    threadgroup ELEM_T* smem_v = smem_k + BLOCK_KV * C_q;                                   \
    threadgroup float*  smem_s = (threadgroup float*)(smem_v + BLOCK_KV * C_v);             \
    threadgroup ELEM_T* smem_p = (threadgroup ELEM_T*)(smem_s + BLOCK_Q * BLOCK_KV);        \
    threadgroup float*  smem_o = (threadgroup float*)(smem_p + BLOCK_Q * BLOCK_KV);         \
    threadgroup float*  smem_m = smem_o + BLOCK_Q * C_v;                                    \
    threadgroup float*  smem_l = smem_m + BLOCK_Q;                                          \
                                                                                            \
    /* Load Q into smem. Invalid rows get zero-padded. */                                   \
    uint q_elements = BLOCK_Q * C_q;                                                        \
    for (uint i = lid; i < q_elements; i += TG_THREADS) {                                   \
        uint row = i / C_q;                                                                 \
        uint col = i % C_q;                                                                 \
        bool valid = row < q_tile_rows;                                                     \
        uint q_row_global = q_start + q_tile_start + row;                                   \
        smem_q[row * C_q + col] = valid                                                     \
            ? q[(q_row_global * H + h) * C_q + col]                                         \
            : (ELEM_T)0;                                                                    \
    }                                                                                       \
    /* Init m, l, O. */                                                                     \
    if (lid < BLOCK_Q) {                                                                    \
        smem_m[lid] = SENTINEL_NEG_INF;                                                     \
        smem_l[lid] = 0.0f;                                                                 \
    }                                                                                       \
    for (uint i = lid; i < BLOCK_Q * C_v; i += TG_THREADS) {                                \
        smem_o[i] = 0.0f;                                                                   \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
                                                                                            \
    /* KV tile loop. */                                                                     \
    for (uint kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += BLOCK_KV) {       \
        uint kv_tile_rows = min((uint)BLOCK_KV, kv_len - kv_tile_start);                    \
                                                                                            \
        /* Cooperative K load [BLOCK_KV, C_q]. */                                           \
        uint k_elements = BLOCK_KV * C_q;                                                   \
        for (uint i = lid; i < k_elements; i += TG_THREADS) {                               \
            uint row = i / C_q;                                                             \
            uint col = i % C_q;                                                             \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_k[row * C_q + col] = valid                                                 \
                ? k[(kv_row_global * H + h) * C_q + col]                                    \
                : (ELEM_T)0;                                                                \
        }                                                                                   \
        /* Cooperative V load [BLOCK_KV, C_v]. */                                           \
        uint v_elements = BLOCK_KV * C_v;                                                   \
        for (uint i = lid; i < v_elements; i += TG_THREADS) {                               \
            uint row = i / C_v;                                                             \
            uint col = i % C_v;                                                             \
            bool valid = row < kv_tile_rows;                                                \
            uint kv_row_global = kv_start + kv_tile_start + row;                            \
            smem_v[row * C_v + col] = valid                                                 \
                ? v[(kv_row_global * H + h) * C_v + col]                                    \
                : (ELEM_T)0;                                                                \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* S = Q @ K^T via simdgroup matmul. */                                             \
        /* Each simdgroup owns 8 Q rows. BLOCK_KV/8 = 4 column tiles. */                    \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            simdgroup_matrix<float, 8, 8> s_acc[BLOCK_KV / 8];                              \
            for (uint t = 0; t < BLOCK_KV / 8; t++) {                                       \
                s_acc[t] = simdgroup_matrix<float, 8, 8>(0.0f);                             \
            }                                                                               \
            for (uint k8 = 0; k8 < C_q; k8 += 8) {                                          \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat, smem_q + simd_id * 8 * C_q + k8, C_q);                \
                for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    /* B = K^T[k8:k8+8, tn*8:tn*8+8] = K[tn*8:tn*8+8, k8:k8+8]^T. */        \
                    simdgroup_load(b_mat, smem_k + tn * 8 * C_q + k8,                       \
                                   C_q, ulong2(0, 0), true);                                \
                    simdgroup_multiply_accumulate(s_acc[tn], a_mat, b_mat, s_acc[tn]);      \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < BLOCK_KV / 8; tn++) {                                    \
                simdgroup_store(s_acc[tn],                                                  \
                                smem_s + simd_id * 8 * BLOCK_KV + tn * 8,                   \
                                BLOCK_KV);                                                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* Per-row online softmax, parallelized 4 threads per row                */         \
        /* (TG_THREADS=64, BLOCK_Q=16 → 4 threads per row).                     */         \
        /*   row       = lid / 4                                                */         \
        /*   col_group = lid % 4 (0..3)                                         */         \
        /* Each thread handles BLOCK_KV/4 = 8 S cols and C_v/4 O cols.          */         \
        /* 4-way intra-row reduction uses simd_shuffle_xor on lanes within the  */         \
        /* same simdgroup (rows 0..7 → simd 0, rows 8..15 → simd 1).            */         \
        {                                                                                   \
            uint row       = lid / 4;                                                       \
            uint col_group = lid % 4;                                                       \
            bool valid_row = row < q_tile_rows;                                             \
                                                                                            \
            uint s_col_start = col_group * (BLOCK_KV / 4);                                  \
            uint s_col_end   = s_col_start + (BLOCK_KV / 4);                                \
                                                                                            \
            float partial_max = SENTINEL_NEG_INF;                                           \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float s_val = (j < kv_tile_rows)                                            \
                    ? smem_s[row * BLOCK_KV + j] * scale                                    \
                    : SENTINEL_NEG_INF;                                                     \
                smem_s[row * BLOCK_KV + j] = s_val;                                         \
                if (s_val > partial_max) partial_max = s_val;                               \
            }                                                                               \
            float row_max = partial_max;                                                    \
            row_max = max(row_max, simd_shuffle_xor(row_max, 1u));                          \
            row_max = max(row_max, simd_shuffle_xor(row_max, 2u));                          \
                                                                                            \
            float m_old = smem_m[row];                                                      \
            float m_new = max(m_old, row_max);                                              \
            float alpha = exp(m_old - m_new);                                               \
                                                                                            \
            float partial_sum = 0.0f;                                                       \
            for (uint j = s_col_start; j < s_col_end; j++) {                                \
                float p_val = exp(smem_s[row * BLOCK_KV + j] - m_new);                      \
                smem_p[row * BLOCK_KV + j] = (ELEM_T)p_val;                                 \
                partial_sum += p_val;                                                       \
            }                                                                               \
            float row_sum = partial_sum;                                                    \
            row_sum += simd_shuffle_xor(row_sum, 1u);                                       \
            row_sum += simd_shuffle_xor(row_sum, 2u);                                       \
                                                                                            \
            if (col_group == 0) {                                                           \
                float l_new = smem_l[row] * alpha + row_sum;                                \
                smem_l[row] = valid_row ? l_new : 0.0f;                                     \
                smem_m[row] = m_new;                                                        \
            }                                                                               \
                                                                                            \
            uint o_cols_per_thread = C_v / 4;                                               \
            uint o_col_start = col_group * o_cols_per_thread;                               \
            for (uint c = 0; c < o_cols_per_thread; c++) {                                  \
                smem_o[row * C_v + o_col_start + c] *= alpha;                               \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
                                                                                            \
        /* O += P @ V via simdgroup matmul. Load o_acc from smem_o (post-rescale). */       \
        if (simd_id < (BLOCK_Q / 8)) {                                                      \
            uint n_tiles_cv = C_v / 8;                                                      \
            simdgroup_matrix<float, 8, 8> o_acc[MAX_HEAD_DIM / 8];                          \
            for (uint tn = 0; tn < n_tiles_cv; tn++) {                                      \
                simdgroup_load(o_acc[tn],                                                   \
                               smem_o + simd_id * 8 * C_v + tn * 8, C_v);                   \
            }                                                                               \
            for (uint k8 = 0; k8 < BLOCK_KV; k8 += 8) {                                     \
                simdgroup_matrix<ELEM_T, 8, 8> a_mat;                                       \
                simdgroup_load(a_mat,                                                       \
                               smem_p + simd_id * 8 * BLOCK_KV + k8, BLOCK_KV);             \
                for (uint tn = 0; tn < n_tiles_cv; tn++) {                                  \
                    simdgroup_matrix<ELEM_T, 8, 8> b_mat;                                   \
                    simdgroup_load(b_mat, smem_v + k8 * C_v + tn * 8, C_v);                 \
                    simdgroup_multiply_accumulate(o_acc[tn], a_mat, b_mat, o_acc[tn]);      \
                }                                                                           \
            }                                                                               \
            for (uint tn = 0; tn < n_tiles_cv; tn++) {                                      \
                simdgroup_store(o_acc[tn],                                                  \
                                smem_o + simd_id * 8 * C_v + tn * 8, C_v);                  \
            }                                                                               \
        }                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    }                                                                                       \
                                                                                            \
    /* Normalize and write output. */                                                       \
    for (uint i = lid; i < BLOCK_Q * C_v; i += TG_THREADS) {                                \
        uint row = i / C_v;                                                                 \
        uint col = i % C_v;                                                                 \
        if (row < q_tile_rows) {                                                            \
            float l_val = smem_l[row];                                                      \
            float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;                           \
            uint q_row_global = q_start + q_tile_start + row;                               \
            out[(q_row_global * H + h) * C_v + col] =                                       \
                (ELEM_T)(smem_o[row * C_v + col] * inv_l);                                  \
        }                                                                                   \
    }                                                                                       \
}

SPARSE_ATTN_FLASH_FWD_KERNEL(sparse_attention_tiled_fwd,        float)
SPARSE_ATTN_FLASH_FWD_KERNEL(sparse_attention_tiled_fwd_half,   half)
SPARSE_ATTN_FLASH_FWD_KERNEL(sparse_attention_tiled_fwd_bfloat, bfloat)
