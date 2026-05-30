#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused variable-length sparse attention forward.
//
// Replaces the SDPA-padded path used on MPS today (pad to [N, max_len, H, C],
// build a [N, 1, Lq, Lkv] mask, invoke F.scaled_dot_product_attention, unpad).
// For trellis2-style sparse batches where max/mean seqlen ratio is often 5-20x,
// the SDPA path spends the majority of its time padding and masking rather
// than on the actual attention math.
//
// Algorithm (online softmax, one pass per Q row):
//   m = -inf; l = 0; o = 0
//   for each kv row:
//       s = dot(q, k) * scale
//       m_new = max(m, s)
//       alpha = exp(m - m_new)
//       l = l * alpha + exp(s - m_new)
//       o = o * alpha + exp(s - m_new) * v
//       m = m_new
//   output = o / l
//
// Grid: (max_q_seqlen, H, N). Each thread handles one (sequence, head, q-row).
// Wasted threads on shorter sequences early-exit; this keeps the kernel simple
// and the wasted work is bounded by max_q_seqlen/mean_q_seqlen, typically
// ~5-10x — still a large win vs SDPA's N^2-in-max-len cost.
//
// Head dimension (C_q, C_v) is runtime-variable; we use per-thread local
// arrays sized for MAX_HEAD_DIM. trellis2 uses head_dim=64 everywhere; the
// ceiling of 128 leaves margin for the common 32/64/128 range.
// ============================================================================

#define MAX_HEAD_DIM 128
#define SENTINEL_NEG_INF -1.0e30f

#define SPARSE_ATTN_FWD_KERNEL(NAME, ELEM_T)                                                \
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
    uint3 tid [[thread_position_in_grid]]                                                   \
) {                                                                                         \
    uint q_idx_in_seq = tid.x;                                                              \
    uint h            = tid.y;                                                              \
    uint seq          = tid.z;                                                              \
                                                                                            \
    if (h >= H) return;                                                                     \
                                                                                            \
    uint q_start = (uint)cu_seqlens_q[seq];                                                 \
    uint q_end   = (uint)cu_seqlens_q[seq + 1];                                             \
    uint q_len   = q_end - q_start;                                                         \
    if (q_idx_in_seq >= q_len) return;                                                      \
                                                                                            \
    uint kv_start = (uint)cu_seqlens_kv[seq];                                               \
    uint kv_end   = (uint)cu_seqlens_kv[seq + 1];                                           \
                                                                                            \
    uint q_row = q_start + q_idx_in_seq;                                                    \
                                                                                            \
    /* Load Q row [C_q] into registers (up to MAX_HEAD_DIM). */                             \
    float q_local[MAX_HEAD_DIM];                                                            \
    for (uint c = 0; c < C_q; c++) {                                                        \
        q_local[c] = (float)q[(q_row * H + h) * C_q + c];                                   \
    }                                                                                       \
                                                                                            \
    /* Online softmax state. */                                                             \
    float m = SENTINEL_NEG_INF;                                                             \
    float l = 0.0f;                                                                         \
    float o_local[MAX_HEAD_DIM];                                                            \
    for (uint c = 0; c < C_v; c++) o_local[c] = 0.0f;                                       \
                                                                                            \
    /* Stream KV rows for this sequence. */                                                 \
    for (uint kv_row = kv_start; kv_row < kv_end; kv_row++) {                               \
        /* s = dot(q, k_row) * scale */                                                     \
        float s = 0.0f;                                                                     \
        for (uint c = 0; c < C_q; c++) {                                                    \
            s += q_local[c] * (float)k[(kv_row * H + h) * C_q + c];                         \
        }                                                                                   \
        s *= scale;                                                                         \
                                                                                            \
        /* m_new = max(m, s); alpha = exp(m - m_new); beta = exp(s - m_new). */             \
        float m_new = max(m, s);                                                            \
        float alpha = exp(m - m_new);                                                       \
        float beta  = exp(s - m_new);                                                       \
                                                                                            \
        /* Scale previous accumulators, add V row weighted by beta. */                      \
        for (uint c = 0; c < C_v; c++) {                                                    \
            o_local[c] = o_local[c] * alpha                                                 \
                        + beta * (float)v[(kv_row * H + h) * C_v + c];                      \
        }                                                                                   \
        l = l * alpha + beta;                                                               \
        m = m_new;                                                                          \
    }                                                                                       \
                                                                                            \
    /* Normalize and write output. Empty KV sequence -> zeros. */                           \
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;                                           \
    for (uint c = 0; c < C_v; c++) {                                                        \
        out[(q_row * H + h) * C_v + c] = (ELEM_T)(o_local[c] * inv_l);                      \
    }                                                                                       \
}

SPARSE_ATTN_FWD_KERNEL(sparse_attention_fwd, float)
SPARSE_ATTN_FWD_KERNEL(sparse_attention_fwd_half, half)
SPARSE_ATTN_FWD_KERNEL(sparse_attention_fwd_bfloat, bfloat)
