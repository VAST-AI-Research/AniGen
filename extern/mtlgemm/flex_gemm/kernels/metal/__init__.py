"""
Metal backend for flex_gemm.

Exports 22 functions:
  - 12 C functions from _C (PyBind11 extension)
  - 10 Python-callable GEMM/weighted-sum functions from _C
"""

import os as _os
import warnings as _warnings


def _metallib_staleness_check():
    """Warn loudly if any .metal source is newer than the shipped metallib —
    pip's editable-install path skips setup.py build_ext, so a `pip install
    -e .` after editing a shader silently reuses the old metallib. Set
    FLEX_GEMM_METALLIB_SKIP_CHECK=1 to silence."""
    if _os.environ.get("FLEX_GEMM_METALLIB_SKIP_CHECK") == "1":
        return
    here = _os.path.dirname(_os.path.abspath(__file__))
    metallib = _os.path.join(here, "flex_gemm.metallib")
    if not _os.path.exists(metallib):
        return  # fresh install, not our business
    lib_m = _os.path.getmtime(metallib)
    newer = []
    for root, _, files in _os.walk(here):
        for f in files:
            if f.endswith((".metal", ".h")):
                p = _os.path.join(root, f)
                if _os.path.getmtime(p) > lib_m + 1:  # 1s slop for fs granularity
                    newer.append(_os.path.relpath(p, here))
    if newer:
        _warnings.warn(
            "flex_gemm.metallib is older than these shader sources:\n  - "
            + "\n  - ".join(newer[:6])
            + ("\n  ...\n" if len(newer) > 6 else "\n")
            + "Rebuild with: cd $(python -c 'import flex_gemm, os; "
            "print(os.path.dirname(flex_gemm.__file__) + \"/..\")')  && "
            "python setup.py build_ext --inplace\n"
            "Or set FLEX_GEMM_METALLIB_SKIP_CHECK=1 to silence.",
            stacklevel=3,
        )


_metallib_staleness_check()

from . import _C

# ============================================================================
# Hash (5 functions) — from _C
# ============================================================================
hashmap_insert_cuda = _C.hashmap_insert_cuda
hashmap_lookup_cuda = _C.hashmap_lookup_cuda
hashmap_insert_3d_cuda = _C.hashmap_insert_3d_cuda
hashmap_lookup_3d_cuda = _C.hashmap_lookup_3d_cuda
hashmap_insert_3d_idx_as_val_cuda = _C.hashmap_insert_3d_idx_as_val_cuda

# ============================================================================
# Serialize (4 functions) — from _C
# ============================================================================
z_order_encode = _C.z_order_encode
z_order_decode = _C.z_order_decode
hilbert_encode = _C.hilbert_encode
hilbert_decode = _C.hilbert_decode

# ============================================================================
# Grid sample (2 functions) — from _C
# ============================================================================
hashmap_build_grid_sample_3d_nearest_neighbor_map = _C.hashmap_build_grid_sample_3d_nearest_neighbor_map
hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight = _C.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight

# ============================================================================
# Spconv neighbor map (3 functions) — from _C
# ============================================================================
hashmap_build_submanifold_conv_neighbour_map_cuda = _C.hashmap_build_submanifold_conv_neighbour_map_cuda
neighbor_map_post_process_for_masked_implicit_gemm_1 = _C.neighbor_map_post_process_for_masked_implicit_gemm_1
neighbor_map_post_process_for_masked_implicit_gemm_2 = _C.neighbor_map_post_process_for_masked_implicit_gemm_2

# ============================================================================
# Weighted sum (2 functions) — from _C
# ============================================================================
def indice_weighed_sum_fwd(input, indices, weight):
    return _C.indice_weighted_sum_fwd(
        input.contiguous(), indices.contiguous(), weight.contiguous()
    )

def indice_weighed_sum_bwd_input(grad_output, indices, weight, N):
    return _C.indice_weighted_sum_bwd_input(
        grad_output.contiguous(), indices.contiguous(), weight.contiguous(), N
    )

# ============================================================================
# Spconv GEMM (8 functions) — Phase 6: Metal compute shader implicit GEMM
# ============================================================================
import torch

def sparse_submanifold_conv_fwd_implicit_gemm(input, weight, bias, neighbor_map):
    return _C.spconv_fwd_implicit_gemm(
        input.contiguous(), weight.contiguous(),
        bias.contiguous() if bias is not None else torch.empty(0),
        neighbor_map.contiguous()
    )

def sparse_submanifold_conv_bwd_implicit_gemm(grad_output, input, weight, bias, neighbor_map):
    grad_input, grad_weight = _C.spconv_bwd_implicit_gemm(
        grad_output.contiguous(), input.contiguous(),
        weight.contiguous(), neighbor_map.contiguous()
    )
    grad_bias = grad_output.sum(dim=0) if bias is not None else None
    return grad_input, grad_weight, grad_bias

# Split-K forward/backward currently map to the plain implicit-GEMM kernel.
# This is a true aliasing, not a silent fallback — the Metal implicit-GEMM
# kernel already has sufficient parallelism at the shapes trellis2 runs.
sparse_submanifold_conv_fwd_implicit_gemm_splitk = sparse_submanifold_conv_fwd_implicit_gemm
sparse_submanifold_conv_bwd_implicit_gemm_splitk = sparse_submanifold_conv_bwd_implicit_gemm

# Masked implicit GEMM calls the real Metal kernels (skips invalid V positions
# per n-block via the precomputed sorted_idx + valid_kernel + valid_kernel_seg
# for fwd and bwd-input; uses valid_signal_* for bwd-weight).

# B1 block size in the masked kernel (must match GEMM_BLOCK_N in metal/config.h).
_MASKED_B1 = 64

def sparse_submanifold_conv_fwd_masked_implicit_gemm(input, weight, bias, neighbor_map,
                                                      sorted_idx, valid_kernel_cb, valid_kernel_seg_cb):
    valid_kernel = valid_kernel_cb(_MASKED_B1)
    valid_kernel_seg = valid_kernel_seg_cb(_MASKED_B1)
    return _C.spconv_fwd_masked_implicit_gemm(
        input.contiguous(), weight.contiguous(),
        bias.contiguous() if bias is not None else torch.empty(0),
        neighbor_map.contiguous(),
        sorted_idx.contiguous(),
        valid_kernel.contiguous(),
        valid_kernel_seg.contiguous(),
    )

def sparse_submanifold_conv_bwd_masked_implicit_gemm(grad_output, input, weight, bias, neighbor_map,
                                                      sorted_idx, valid_kernel_cb, valid_kernel_seg_cb,
                                                      valid_signal_i, valid_signal_o, valid_signal_seg):
    valid_kernel = valid_kernel_cb(_MASKED_B1)
    valid_kernel_seg = valid_kernel_seg_cb(_MASKED_B1)
    grad_input, grad_weight = _C.spconv_bwd_masked_implicit_gemm(
        grad_output.contiguous(), input.contiguous(),
        weight.contiguous(), neighbor_map.contiguous(),
        sorted_idx.contiguous(),
        valid_kernel.contiguous(),
        valid_kernel_seg.contiguous(),
        valid_signal_i.contiguous(),
        valid_signal_o.contiguous(),
        valid_signal_seg.contiguous(),
    )
    grad_bias = grad_output.sum(dim=0) if bias is not None else None
    return grad_input, grad_weight, grad_bias

sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk = sparse_submanifold_conv_fwd_masked_implicit_gemm
sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk = sparse_submanifold_conv_bwd_masked_implicit_gemm


# ============================================================================
# Sparse attention (2 functions) — fused variable-length attention fwd + bwd
# ============================================================================

def sparse_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                         max_q_seqlen, max_kv_seqlen, scale):
    """Fused variable-length sparse attention forward.

    Args:
        q: [T_q, H, C_q] packed fp32/fp16/bf16 tensor.
        k: [T_kv, H, C_q] packed tensor (same dtype as q).
        v: [T_kv, H, C_v] packed tensor.
        cu_seqlens_q: [N+1] int32 prefix sum of per-sequence q lengths.
        cu_seqlens_kv: [N+1] int32 prefix sum of per-sequence kv lengths.
        max_q_seqlen, max_kv_seqlen: ints, per-batch maxima.
        scale: float, typically 1/sqrt(C_q).

    Returns:
        [T_q, H, C_v] tensor, same dtype + device as q.
    """
    return _C.sparse_attention_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_kv.contiguous(),
        int(max_q_seqlen), int(max_kv_seqlen), float(scale),
    )


def sparse_attention_bwd(q, k, v, d_out, cu_seqlens_q, cu_seqlens_kv,
                         max_q_seqlen, max_kv_seqlen, scale):
    """Fused variable-length sparse attention backward.

    Two-pass naive implementation: dispatches `sparse_attention_bwd_q` (one
    thread per Q row, recomputes m, l and D, writes dQ) then
    `sparse_attention_bwd_kv` (one thread per KV row, uses saved m, l, D to
    accumulate dK, dV).

    Args:
        q, k, v, cu_seqlens_*, max_*, scale: as in sparse_attention_fwd.
        d_out: [T_q, H, C_v] gradient w.r.t. the forward output.

    Returns:
        (dQ [T_q, H, C_q], dK [T_kv, H, C_q], dV [T_kv, H, C_v]).
    """
    return _C.sparse_attention_bwd(
        q.contiguous(), k.contiguous(), v.contiguous(), d_out.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_kv.contiguous(),
        int(max_q_seqlen), int(max_kv_seqlen), float(scale),
    )
