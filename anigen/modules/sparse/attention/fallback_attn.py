"""Naive fp32 variable-length attention for MPS (replaces flash-attn varlen).

Contract matches anigen.modules.sparse.attention.full_attn: q is [Tq, H, C],
k/v are [Tkv, H, C], with block-diagonal (per-batch) attention defined by the
q_seqlen / kv_seqlen segment lengths. Computed segment-by-segment in fp32 to
avoid the MPS fused-SDPA cliff. Output is [Tq, H, Cv].
"""
import math
from typing import List
import torch

# Query rows are processed in tiles of this size against the full key/value
# segment, mirroring Pixal3D's _NA2D_QUERY_CHUNK approach. This keeps the
# attention-weight matrix bounded ([H, _QUERY_CHUNK, sk]) so very long segments
# (the HR sparse stages) do not blow up MPS/CPU memory. The math is identical to
# computing the whole segment at once.
_QUERY_CHUNK = 2048


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
    out = torch.empty(q.shape[0], H, Cv, device=q.device, dtype=q.dtype)
    qo = ko = 0
    for sq, sk in zip(q_seqlen, kv_seqlen):
        ki = k[ko:ko + sk].permute(1, 0, 2).float()          # [H, sk, C]
        vi = v[ko:ko + sk].permute(1, 0, 2).float()          # [H, sk, Cv]
        for c0 in range(0, sq, _QUERY_CHUNK):
            c1 = min(c0 + _QUERY_CHUNK, sq)
            qi = q[qo + c0:qo + c1].permute(1, 0, 2).float()  # [H, chunk, C]
            scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            seg = torch.matmul(weights, vi).permute(1, 0, 2)  # [chunk, H, Cv]
            out[qo + c0:qo + c1] = seg.to(out.dtype)
        qo += sq; ko += sk
    return out
