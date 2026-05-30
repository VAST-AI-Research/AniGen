from typing import *
import torch
from .. import kernels


@torch.no_grad()
def encode_seq(
    coords: torch.Tensor,
    shape: torch.Size,
    mode: Literal['z_order', 'hilbert'] = 'z_order'
) -> torch.Tensor:
    """
    Encodes 3D coordinates into a code.

    Args:
        coords: a tensor of shape [N, 4] containing the 3D coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        mode: the encoding mode to use.
    """
    assert coords.shape[-1] == 4 and coords.ndim == 2, "Input coordinates must be of shape [N, 4]"
    N, C, H, W, D = shape
    
    max_coord = max(H, W, D)
    bit_length = max_coord.bit_length()
    batch_bit_length = N.bit_length()
    total_bit_length = batch_bit_length + bit_length * 3
    
    if total_bit_length <= 32:
        codes = torch.empty(coords.shape[0], dtype=torch.int32, device=coords.device)
    elif total_bit_length <= 64:
        codes = torch.empty(coords.shape[0], dtype=torch.int64, device=coords.device)
    else:
        raise ValueError(f"Spatial dimensions are too large: {shape}")
    
    if mode == 'z_order':
        kernels.cuda.z_order_encode(coords, bit_length, codes)
    elif mode == 'hilbert':
        kernels.cuda.hilbert_encode(coords, bit_length, codes)
    else:
        raise ValueError(f"Unknown encoding mode: {mode}")
    
    return codes


def decode_seq(
    code: torch.Tensor,
    shape: torch.Size,
    mode: Literal['z_order', 'hilbert'] = 'z_order'
) -> torch.Tensor:
    """
    Decodes a code into 3D coordinates.

    Args:
        code: a tensor of shape [N] containing the code.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        mode: the decoding mode to use.
    """
    assert code.ndim == 1, "Input code must be of shape [N]"
    N, C, H, W, D = shape
    max_coord = max(H, W, D)
    bit_length = max_coord.bit_length()
    
    if mode == 'z_order':
        coords = kernels.cuda.z_order_decode(code, bit_length)
    elif mode == 'hilbert':
        coords = kernels.cuda.hilbert_decode(code, bit_length)
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    
    return coords
