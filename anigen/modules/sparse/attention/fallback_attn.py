"""Naive fp32 variable-length attention for MPS (replaces flash-attn varlen).

Contract matches anigen.modules.sparse.attention.full_attn: q is [Tq, H, C],
k/v are [Tkv, H, C], with block-diagonal (per-batch) attention defined by the
q_seqlen / kv_seqlen segment lengths. Computed segment-by-segment in fp32 to
avoid the MPS fused-SDPA cliff. Output is [Tq, H, Cv].
"""
import math
import os
from typing import List
import torch

# Query rows are processed in tiles of this size against the key/value segment.
# This keeps the attention-weight matrix bounded so very long segments (the HR
# sparse stages) do not blow up MPS/CPU memory. The math is identical to
# computing the whole segment at once.
_QUERY_CHUNK = int(os.environ.get("ANIGEN_ATTN_QUERY_CHUNK", "2048"))
# On MPS a score tensor of [H, q_chunk, sk] with very large sk (the full SLAT
# token set, tens of thousands) becomes multi-GB and copies/softmaxes
# pathologically slowly. When set, we additionally tile over the KEY dimension
# and combine partial results with an online (streaming) softmax so peak memory
# is bounded by [H, q_chunk, _KEY_CHUNK] regardless of segment length. Disabled
# by default (==0) so CUDA / small-segment behavior is byte-for-byte unchanged.
_KEY_CHUNK = int(os.environ.get("ANIGEN_ATTN_KEY_CHUNK", "0"))
_DEBUG = os.environ.get("ANIGEN_ATTN_DEBUG", "0") == "1"


def naive_varlen_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_seqlen: List[int], kv_seqlen: List[int],
) -> torch.Tensor:
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    H, C = q.shape[1], q.shape[2]
    Cv = v.shape[2]
    scale = 1.0 / math.sqrt(C)
    assert sum(q_seqlen) == q.shape[0] and sum(kv_seqlen) == k.shape[0], \
        "naive_varlen_attention: seqlens do not cover all tokens"
    if _DEBUG:
        print(f"[naive_varlen_attention] H={H} C={C} Cv={Cv} "
              f"segments={len(q_seqlen)} max_q={max(q_seqlen)} max_kv={max(kv_seqlen)} "
              f"q_chunk={_QUERY_CHUNK} key_chunk={_KEY_CHUNK}", flush=True)
    out = torch.empty(q.shape[0], H, Cv, device=q.device, dtype=q.dtype)
    qo = ko = 0
    for sq, sk in zip(q_seqlen, kv_seqlen):
        ki = k[ko:ko + sk].permute(1, 0, 2).float()          # [H, sk, C]
        vi = v[ko:ko + sk].permute(1, 0, 2).float()          # [H, sk, Cv]
        for c0 in range(0, sq, _QUERY_CHUNK):
            c1 = min(c0 + _QUERY_CHUNK, sq)
            qi = q[qo + c0:qo + c1].permute(1, 0, 2).float()  # [H, chunk, C]
            if _KEY_CHUNK <= 0 or sk <= _KEY_CHUNK:
                scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
                weights = torch.softmax(scores, dim=-1)
                seg = torch.matmul(weights, vi).permute(1, 0, 2)  # [chunk, H, Cv]
            else:
                # Online (streaming) softmax over key tiles: bounds peak memory to
                # [H, q_chunk, _KEY_CHUNK] instead of [H, q_chunk, sk]. Mathematically
                # identical to the full softmax (running max + rescaled accumulators).
                Hh, qn = qi.shape[0], qi.shape[1]
                m_run = torch.full((Hh, qn, 1), float('-inf'), device=qi.device, dtype=qi.dtype)
                l_run = torch.zeros((Hh, qn, 1), device=qi.device, dtype=qi.dtype)
                acc = torch.zeros((Hh, qn, Cv), device=qi.device, dtype=qi.dtype)
                for k0 in range(0, sk, _KEY_CHUNK):
                    k1 = min(k0 + _KEY_CHUNK, sk)
                    kt = ki[:, k0:k1, :]                       # [H, kt, C]
                    vt = vi[:, k0:k1, :]                       # [H, kt, Cv]
                    s = torch.matmul(qi, kt.transpose(-2, -1)) * scale  # [H, qn, kt]
                    m_tile = s.amax(dim=-1, keepdim=True)      # [H, qn, 1]
                    m_new = torch.maximum(m_run, m_tile)
                    p = torch.exp(s - m_new)                   # [H, qn, kt]
                    alpha = torch.exp(m_run - m_new)           # rescale prior accum
                    l_run = l_run * alpha + p.sum(dim=-1, keepdim=True)
                    acc = acc * alpha + torch.matmul(p, vt)
                    m_run = m_new
                seg = (acc / l_run).permute(1, 0, 2)           # [chunk, H, Cv]
            out[qo + c0:qo + c1] = seg.to(out.dtype)
        qo += sq; ko += sk
    return out
