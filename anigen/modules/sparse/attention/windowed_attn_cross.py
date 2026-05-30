from typing import *
import torch
import math
from .. import SparseTensor
from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
elif ATTN == 'naive':
    from .fallback_attn import naive_varlen_attention
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_windowed_scaled_dot_product_cross_attention',
]


def calc_window_partition_cross(
    tensor: SparseTensor,
    context: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """

    def calc_window_partition_(tensor, window_size, shift_window):
        DIM = tensor.coords.shape[1] - 1
        shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
        window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
        shifted_coords = tensor.coords.clone().detach()
        shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

        MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
        NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
        OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

        shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
        shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
        fwd_indices = torch.argsort(shifted_indices)
        bwd_indices = torch.empty_like(fwd_indices)
        bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
        seq_lens = torch.bincount(shifted_indices)
        seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32) // OFFSET[0]
        return fwd_indices, bwd_indices, seq_lens, seq_batch_indices

    fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition_(tensor, window_size, shift_window)
    fwd_indices_context, bwd_indices_context, seq_lens_context, seq_batch_indices_context = calc_window_partition_(context, window_size, shift_window)
    # Pad the shorter one to the shape of the other with 0 tail
    max_len = max(seq_lens.shape[0], seq_lens_context.shape[0])
    if seq_lens.shape[0] < max_len:
        pad_size = max_len - seq_lens.shape[0]
        seq_lens = torch.cat([seq_lens, torch.zeros(pad_size, dtype=seq_lens.dtype, device=seq_lens.device)])
    if seq_lens_context.shape[0] < max_len:
        pad_size = max_len - seq_lens_context.shape[0]
        seq_lens_context = torch.cat([seq_lens_context, torch.zeros(pad_size, dtype=seq_lens_context.dtype, device=seq_lens_context.device)])
    mask = (seq_lens != 0) | (seq_lens_context != 0)
    seq_lens = seq_lens[mask].tolist()
    seq_lens_context = seq_lens_context[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, fwd_indices_context, bwd_indices_context, seq_lens_context
    

def sparse_windowed_scaled_dot_product_cross_attention(
    q: SparseTensor,
    kv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0),
    cache_suffix: str = '',
) -> SparseTensor:
    """
    Apply windowed scaled dot product cross attention to a sparse tensor.

    Args:
        q, kv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """
    assert len(q.shape)  == 4 and q.shape[1]  == 1, f"Invalid shape for q, got {q.shape}, expected [N, *, 1, H, C]"
    assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"

    serialization_spatial_cache_name_q = f'window_partition_{window_size}_{shift_window}_cross_q' + cache_suffix
    serialization_spatial_cache_q = q.get_spatial_cache(serialization_spatial_cache_name_q)
    serialization_spatial_cache_name_kv = f'window_partition_{window_size}_{shift_window}_cross_kv' + cache_suffix
    serialization_spatial_cache_kv = kv.get_spatial_cache(serialization_spatial_cache_name_kv)
    if serialization_spatial_cache_q is None or serialization_spatial_cache_kv is None:
        q_fwd_indices, q_bwd_indices, q_seq_lens, kv_fwd_indices, kv_bwd_indices, kv_seq_lens = calc_window_partition_cross(q, kv, window_size, shift_window)
        q.register_spatial_cache(serialization_spatial_cache_name_q, (q_fwd_indices, q_bwd_indices, q_seq_lens))
        kv.register_spatial_cache(serialization_spatial_cache_name_kv, (kv_fwd_indices, kv_bwd_indices, kv_seq_lens))
    else:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens = serialization_spatial_cache_kv
        q_fwd_indices, q_bwd_indices, q_seq_lens = serialization_spatial_cache_q

    M_q,  T_q,  H_q,  C_q  = q_fwd_indices.shape[0],  q.feats.shape[0],  q.feats.shape[2],  q.feats.shape[3]
    M_kv, T_kv, H_kv, C_kv = kv_fwd_indices.shape[0], kv.feats.shape[0], kv.feats.shape[2], kv.feats.shape[3]
    assert (H_q == H_kv and C_q == C_kv), \
        f"Mismatch in shapes: q ({M_q}, {T_q}, {H_q}, {C_q}), kv ({M_kv}, {T_kv}, {H_kv}, {C_kv})"
    
    q_feats  = q.feats[q_fwd_indices]        # [M, 1, H, C]
    kv_feats = kv.feats[kv_fwd_indices]      # [M, 2, H, C]

    if ATTN == 'xformers':
        q, k, v = q_feats[:, 0], kv_feats.unbind(dim=1)         # [M, H, C]
        q = q.unsqueeze(0)                                      # [1, M, H, C]
        k = k.unsqueeze(0)                                      # [1, M, H, C]
        v = v.unsqueeze(0)                                      # [1, M, H, C]
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seq_lens, kv_seq_lens)
        out = xops.memory_efficient_attention(q, k, v, mask)[0] # [M, H, C]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seq_lens), dim=0)],  dim=0).to(q.device).int()
        cu_seqlens_k = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seq_lens), dim=0)], dim=0).to(kv.device).int()
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats[:, 0], kv_feats, cu_seqlens_q, cu_seqlens_k, max(q_seq_lens), max(kv_seq_lens)) # [M, H, C]
    elif ATTN == 'naive':
        nq = q_feats[:, 0]                                          # [M, H, C]
        nk, nv = kv_feats.unbind(dim=1)                            # [M, H, C]
        out = naive_varlen_attention(nq, nk, nv, q_seq_lens, kv_seq_lens)  # [M, H, C]

    out = out[q_bwd_indices]      # [T, H, C]
    return q.replace(out)

