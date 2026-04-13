"""
Geodesic smooth noise utility for skin-branch noise in SLat flow models.

Smooths only the skin-branch channels of a SparseTensor noise using
6-neighbor voxel adjacency (NOT convolution-based).
"""

from typing import Optional
import torch
from ..modules import sparse as sp


# 6 axis-aligned neighbor offsets (face-adjacent voxels only)
_OFFSETS_6 = torch.tensor([
    [-1,  0,  0],
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  1,  0],
    [ 0,  0, -1],
    [ 0,  0,  1],
], dtype=torch.int32)


def _build_neighbor_indices(coords: torch.Tensor):
    """
    Build neighbour lookup from sparse voxel coordinates.

    Args:
        coords: (N, 4) int tensor – columns are [batch, x, y, z].

    Returns:
        src: (E,) long tensor – source voxel indices for each neighbour edge.
        dst: (E,) long tensor – destination (neighbour) voxel indices.
    """
    device = coords.device
    coords = coords.long()

    # Pack (batch, x, y, z) into a unique int64 key.
    # Coords are in [0, 1023] per the SparseTensor doc, batch index is small.
    M = 2048
    keys = coords[:, 0] * (M ** 3) + coords[:, 1] * (M ** 2) + coords[:, 2] * M + coords[:, 3]

    # Sort keys for searchsorted
    sorted_keys, sort_idx = torch.sort(keys)
    inv_sort = torch.empty_like(sort_idx)
    inv_sort[sort_idx] = torch.arange(len(sort_idx), device=device)

    offsets = _OFFSETS_6.to(device)  # (6, 3)

    src_list = []
    dst_list = []
    for off in offsets:
        shifted = coords.clone()
        shifted[:, 1:4] += off
        shifted_keys = shifted[:, 0] * (M ** 3) + shifted[:, 1] * (M ** 2) + shifted[:, 2] * M + shifted[:, 3]

        # Find matches via searchsorted on the sorted key array
        insert_pos = torch.searchsorted(sorted_keys, shifted_keys)
        insert_pos = insert_pos.clamp(max=len(sorted_keys) - 1)
        found = sorted_keys[insert_pos] == shifted_keys

        src_idx = torch.arange(len(coords), device=device)[found]
        dst_idx = sort_idx[insert_pos[found]]

        src_list.append(src_idx)
        dst_list.append(dst_idx)

    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    return src, dst


def smooth_sparse_noise_channels(
    noise_sp: sp.SparseTensor,
    start_ch: int,
    end_ch: int,
    iters: int,
    alpha: float,
) -> sp.SparseTensor:
    """
    Iteratively smooth a channel slice of a SparseTensor's features using
    6-neighbour voxel adjacency.

    Args:
        noise_sp: Input sparse tensor with noise features.
        start_ch: Start index of channels to smooth (inclusive).
        end_ch: End index of channels to smooth (exclusive).
        iters: Number of smoothing iterations.
        alpha: Blending weight – new = (1 - alpha) * old + alpha * neighbour_mean.

    Returns:
        A new SparseTensor with smoothed target channels and re-normalised variance.
    """
    if iters <= 0 or alpha <= 0:
        return noise_sp

    feats = noise_sp.feats.clone()
    coords = noise_sp.coords

    src, dst = _build_neighbor_indices(coords)

    for _ in range(iters):
        target = feats[:, start_ch:end_ch]

        # Scatter-add neighbour features and count neighbours per voxel
        neighbor_sum = torch.zeros_like(target)
        neighbor_count = torch.zeros(target.shape[0], 1, device=target.device, dtype=target.dtype)
        neighbor_sum.index_add_(0, src, target[dst])
        neighbor_count.index_add_(0, src, torch.ones(dst.shape[0], 1, device=target.device, dtype=target.dtype))

        # Avoid division by zero for isolated voxels (no neighbours)
        has_neighbor = neighbor_count > 0
        neighbor_mean = torch.where(has_neighbor, neighbor_sum / neighbor_count, target)

        feats[:, start_ch:end_ch] = (1 - alpha) * target + alpha * neighbor_mean

    # Re-normalise smoothed channels to unit variance per-channel
    smoothed = feats[:, start_ch:end_ch]
    std = smoothed.std(dim=0, keepdim=True).clamp(min=1e-8)
    feats[:, start_ch:end_ch] = smoothed / std

    return noise_sp.replace(feats)


def maybe_geodesic_smooth_slat_noise(
    noise,
    model,
    enabled: bool = False,
    iters: int = 0,
    alpha: float = 0.7,
):
    """
    Conditionally apply geodesic smoothing to the skin-branch channels of SLat noise.

    Early-returns the noise unchanged if:
      - disabled
      - input is not a SparseTensor
      - model lacks in_channels / in_channels_vert_skin
      - channel slice is invalid

    Args:
        noise: The noise (expected to be an sp.SparseTensor for SLat models).
        model: The denoiser model (needs .in_channels and .in_channels_vert_skin).
        enabled: Whether geodesic smoothing is enabled.
        iters: Number of smoothing iterations.
        alpha: Blending weight.

    Returns:
        The (possibly smoothed) noise.
    """
    if not enabled:
        return noise

    if not isinstance(noise, sp.SparseTensor):
        return noise

    in_ch = getattr(model, 'in_channels', None)
    in_ch_skin = getattr(model, 'in_channels_vert_skin', None)
    if in_ch is None or in_ch_skin is None:
        return noise

    start_ch = in_ch
    end_ch = in_ch + in_ch_skin
    total_ch = noise.feats.shape[-1] if noise.feats.dim() > 0 else 0
    if start_ch >= end_ch or end_ch > total_ch:
        return noise

    return smooth_sparse_noise_channels(noise, start_ch, end_ch, iters, alpha)
