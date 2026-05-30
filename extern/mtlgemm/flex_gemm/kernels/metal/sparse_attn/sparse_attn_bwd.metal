#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Variable-length sparse attention backward (naive, correctness-first).
//
// Two kernels, dispatched sequentially:
//
//   bwd_q : grid (max_q_seqlen, H, N) — one thread per Q row.
//           Recomputes (m, l) online, then D = sum_j P_j dP_j, then dQ.
//           Writes dQ and temp buffers m_aux, l_aux, d_aux for bwd_kv.
//
//   bwd_kv: grid (max_kv_seqlen, H, N) — one thread per KV row.
//           Reads m_aux, l_aux, d_aux. For each Q row recomputes P, dP,
//           dS on the fly, accumulates dV = sum_q P dO, dK = sum_q dS Q scale.
//
// Math (standard attention backward):
//   P  = softmax(Q K^T * scale)
//   O  = P V
//   dV = P^T @ dO
//   dP = dO @ V^T
//   D  = rowsum(P * dP)  (per Q row)
//   dS = P * (dP - D)
//   dQ = dS @ K * scale
//   dK = dS^T @ Q * scale
//
// Per-thread compute is O(kv_len) (bwd_q) or O(q_len) (bwd_kv). No smem, no
// simdgroup matmul — the follow-up is a tiled flash-attn-v2-bwd, tracked
// in FOLLOWUPS.md.
// ============================================================================

#define BWD_MAX_HEAD_DIM 128
#define SENTINEL_NEG_INF -1.0e30f

#define SPARSE_ATTN_BWD_Q_KERNEL(NAME, ELEM_T)                                              \
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
    uint3 tid [[thread_position_in_grid]]                                                   \
) {                                                                                         \
    uint q_idx_in_seq = tid.x;                                                              \
    uint h            = tid.y;                                                              \
    uint seq          = tid.z;                                                              \
    if (h >= H) return;                                                                     \
                                                                                            \
    uint q_start = (uint)cu_seqlens_q[seq];                                                 \
    uint q_end   = (uint)cu_seqlens_q[seq + 1];                                             \
    if (q_idx_in_seq >= (q_end - q_start)) return;                                          \
    uint q_row = q_start + q_idx_in_seq;                                                    \
                                                                                            \
    uint kv_start = (uint)cu_seqlens_kv[seq];                                               \
    uint kv_end   = (uint)cu_seqlens_kv[seq + 1];                                           \
                                                                                            \
    float q_local[BWD_MAX_HEAD_DIM];                                                        \
    for (uint c = 0; c < C_q; c++) {                                                        \
        q_local[c] = (float)q[(q_row * H + h) * C_q + c];                                   \
    }                                                                                       \
    float do_local[BWD_MAX_HEAD_DIM];                                                       \
    for (uint c = 0; c < C_v; c++) {                                                        \
        do_local[c] = (float)d_out[(q_row * H + h) * C_v + c];                              \
    }                                                                                       \
                                                                                            \
    /* Pass 1: online softmax (m, l). */                                                    \
    float m = SENTINEL_NEG_INF, l = 0.0f;                                                   \
    for (uint kv = kv_start; kv < kv_end; kv++) {                                           \
        float s = 0.0f;                                                                     \
        for (uint c = 0; c < C_q; c++) {                                                    \
            s += q_local[c] * (float)k[(kv * H + h) * C_q + c];                             \
        }                                                                                   \
        s *= scale;                                                                         \
        float m_new = max(m, s);                                                            \
        float alpha = exp(m - m_new);                                                       \
        float beta  = exp(s - m_new);                                                       \
        l = l * alpha + beta;                                                               \
        m = m_new;                                                                          \
    }                                                                                       \
                                                                                            \
    float l_safe = (l > 0.0f) ? l : 1.0f;                                                   \
                                                                                            \
    /* Pass 2: D = sum_j P_j * dP_j. */                                                     \
    float D = 0.0f;                                                                         \
    for (uint kv = kv_start; kv < kv_end; kv++) {                                           \
        float s = 0.0f;                                                                     \
        for (uint c = 0; c < C_q; c++) {                                                    \
            s += q_local[c] * (float)k[(kv * H + h) * C_q + c];                             \
        }                                                                                   \
        s *= scale;                                                                         \
        float p = exp(s - m) / l_safe;                                                      \
        float dp = 0.0f;                                                                    \
        for (uint c = 0; c < C_v; c++) {                                                    \
            dp += do_local[c] * (float)v[(kv * H + h) * C_v + c];                           \
        }                                                                                   \
        D += p * dp;                                                                        \
    }                                                                                       \
                                                                                            \
    /* Pass 3: dQ = sum_j dS_j * K_j * scale. */                                            \
    float dq_local[BWD_MAX_HEAD_DIM];                                                       \
    for (uint c = 0; c < C_q; c++) dq_local[c] = 0.0f;                                      \
    for (uint kv = kv_start; kv < kv_end; kv++) {                                           \
        float s = 0.0f;                                                                     \
        for (uint c = 0; c < C_q; c++) {                                                    \
            s += q_local[c] * (float)k[(kv * H + h) * C_q + c];                             \
        }                                                                                   \
        s *= scale;                                                                         \
        float p = exp(s - m) / l_safe;                                                      \
        float dp = 0.0f;                                                                    \
        for (uint c = 0; c < C_v; c++) {                                                    \
            dp += do_local[c] * (float)v[(kv * H + h) * C_v + c];                           \
        }                                                                                   \
        float ds = p * (dp - D);                                                            \
        for (uint c = 0; c < C_q; c++) {                                                    \
            dq_local[c] += ds * (float)k[(kv * H + h) * C_q + c] * scale;                   \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    for (uint c = 0; c < C_q; c++) {                                                        \
        d_q[(q_row * H + h) * C_q + c] = (ELEM_T)dq_local[c];                               \
    }                                                                                       \
    m_aux[q_row * H + h] = m;                                                               \
    l_aux[q_row * H + h] = l;                                                               \
    d_aux[q_row * H + h] = D;                                                               \
}

SPARSE_ATTN_BWD_Q_KERNEL(sparse_attention_bwd_q,        float)
SPARSE_ATTN_BWD_Q_KERNEL(sparse_attention_bwd_q_half,   half)
SPARSE_ATTN_BWD_Q_KERNEL(sparse_attention_bwd_q_bfloat, bfloat)


#define SPARSE_ATTN_BWD_KV_KERNEL(NAME, ELEM_T)                                             \
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
    uint3 tid [[thread_position_in_grid]]                                                   \
) {                                                                                         \
    uint kv_idx_in_seq = tid.x;                                                             \
    uint h             = tid.y;                                                             \
    uint seq           = tid.z;                                                             \
    if (h >= H) return;                                                                     \
                                                                                            \
    uint kv_start = (uint)cu_seqlens_kv[seq];                                               \
    uint kv_end   = (uint)cu_seqlens_kv[seq + 1];                                           \
    if (kv_idx_in_seq >= (kv_end - kv_start)) return;                                       \
    uint kv_row = kv_start + kv_idx_in_seq;                                                 \
                                                                                            \
    uint q_start = (uint)cu_seqlens_q[seq];                                                 \
    uint q_end   = (uint)cu_seqlens_q[seq + 1];                                             \
                                                                                            \
    float k_local[BWD_MAX_HEAD_DIM];                                                        \
    for (uint c = 0; c < C_q; c++) {                                                        \
        k_local[c] = (float)k[(kv_row * H + h) * C_q + c];                                  \
    }                                                                                       \
    float v_local[BWD_MAX_HEAD_DIM];                                                        \
    for (uint c = 0; c < C_v; c++) {                                                        \
        v_local[c] = (float)v[(kv_row * H + h) * C_v + c];                                  \
    }                                                                                       \
                                                                                            \
    float dk_local[BWD_MAX_HEAD_DIM];                                                       \
    float dv_local[BWD_MAX_HEAD_DIM];                                                       \
    for (uint c = 0; c < C_q; c++) dk_local[c] = 0.0f;                                      \
    for (uint c = 0; c < C_v; c++) dv_local[c] = 0.0f;                                      \
                                                                                            \
    for (uint qr = q_start; qr < q_end; qr++) {                                             \
        float m = m_aux[qr * H + h];                                                        \
        float l = l_aux[qr * H + h];                                                        \
        float D = d_aux[qr * H + h];                                                        \
        float l_safe = (l > 0.0f) ? l : 1.0f;                                               \
                                                                                            \
        float s = 0.0f;                                                                     \
        for (uint c = 0; c < C_q; c++) {                                                    \
            s += (float)q[(qr * H + h) * C_q + c] * k_local[c];                             \
        }                                                                                   \
        s *= scale;                                                                         \
        float p = exp(s - m) / l_safe;                                                      \
                                                                                            \
        float dp = 0.0f;                                                                    \
        for (uint c = 0; c < C_v; c++) {                                                    \
            dp += (float)d_out[(qr * H + h) * C_v + c] * v_local[c];                        \
        }                                                                                   \
        float ds = p * (dp - D);                                                            \
                                                                                            \
        for (uint c = 0; c < C_v; c++) {                                                    \
            dv_local[c] += p * (float)d_out[(qr * H + h) * C_v + c];                        \
        }                                                                                   \
        for (uint c = 0; c < C_q; c++) {                                                    \
            dk_local[c] += ds * (float)q[(qr * H + h) * C_q + c] * scale;                   \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    for (uint c = 0; c < C_q; c++) {                                                        \
        d_k[(kv_row * H + h) * C_q + c] = (ELEM_T)dk_local[c];                              \
    }                                                                                       \
    for (uint c = 0; c < C_v; c++) {                                                        \
        d_v[(kv_row * H + h) * C_v + c] = (ELEM_T)dv_local[c];                              \
    }                                                                                       \
}

SPARSE_ATTN_BWD_KV_KERNEL(sparse_attention_bwd_kv,        float)
SPARSE_ATTN_BWD_KV_KERNEL(sparse_attention_bwd_kv_half,   half)
SPARSE_ATTN_BWD_KV_KERNEL(sparse_attention_bwd_kv_bfloat, bfloat)
