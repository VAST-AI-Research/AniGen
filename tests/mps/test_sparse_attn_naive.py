import math, torch
from anigen.modules.sparse.attention.fallback_attn import naive_varlen_attention

def _ref_blockdiag(q, k, v, q_seqlen, kv_seqlen):
    outs = []
    qo = ko = 0
    for sq, sk in zip(q_seqlen, kv_seqlen):
        qi = q[qo:qo+sq].permute(1, 0, 2)            # [H, sq, C]
        ki = k[ko:ko+sk].permute(1, 0, 2)            # [H, sk, C]
        vi = v[ko:ko+sk].permute(1, 0, 2)            # [H, sk, C]
        w = torch.softmax(qi @ ki.transpose(-2, -1) / math.sqrt(qi.shape[-1]), dim=-1)
        outs.append((w @ vi).permute(1, 0, 2))       # [sq, H, C]
        qo += sq; ko += sk
    return torch.cat(outs, dim=0)

def test_naive_varlen_matches_reference():
    H, C = 4, 16
    q_seqlen = [30, 50]; kv_seqlen = [30, 50]
    q = torch.randn(sum(q_seqlen), H, C)
    k = torch.randn(sum(kv_seqlen), H, C)
    v = torch.randn(sum(kv_seqlen), H, C)
    out = naive_varlen_attention(q, k, v, q_seqlen, kv_seqlen)
    ref = _ref_blockdiag(q, k, v, q_seqlen, kv_seqlen)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()

def test_naive_varlen_large_token_count():
    H, C = 8, 64
    q_seqlen = kv_seqlen = [21504]   # exceeds the ~18-20k MPS SDPA cliff
    q = torch.randn(21504, H, C); k = torch.randn(21504, H, C); v = torch.randn(21504, H, C)
    out = naive_varlen_attention(q, k, v, q_seqlen, kv_seqlen)
    assert torch.isfinite(out).all()
    assert out.abs().mean() > 0
