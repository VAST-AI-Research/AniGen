"""flex_gemm (Metal) submanifold sparse-conv backend, drop-in for conv_spconv.

Matches ``conv_spconv.SparseConv3d``'s ``nn.Module`` interface but routes the
kernel to ``flex_gemm.ops.spconv.sparse_submanifold_conv3d`` so AniGen's sparse
convolutions run on Apple Silicon (MPS) instead of CUDA-only spconv.

Coverage gate: flex_gemm exposes **only** submanifold conv (stride==1,
``padding is None``). It has no strided ``SparseConv3d`` and no
``SparseInverseConv3d``. Those paths raise ``NotImplementedError`` loudly rather
than silently producing wrong results.
"""
import math
import torch
import torch.nn as nn
import flex_gemm
from flex_gemm.ops.spconv import (
    Algorithm,
    sparse_submanifold_conv3d,
    set_algorithm,
    set_hashmap_ratio,
)
from .. import SparseTensor

__all__ = ['SparseConv3d', 'SparseInverseConv3d']


def _spatial_shape(x: SparseTensor):
    """Spatial extent (W, H, D) of the sparse grid.

    AniGen's ``SparseTensor`` (unlike Pixal3D's) has no ``spatial_shape``
    property. The spconv data holder exposes ``.data.spatial_shape``, but on
    Apple Silicon spconv is not installed, so we derive the extent from the
    coordinates directly. Coords are ``[N, 4]`` with column 0 = batch index.
    """
    data_shape = getattr(getattr(x, "data", None), "spatial_shape", None)
    if isinstance(data_shape, (list, tuple)) and len(data_shape) == 3:
        return [int(s) for s in data_shape]
    coords = x.coords
    # Reduce on CPU: MPS max/amax over int coords can hang the Metal command
    # buffer, and the result is needed host-side anyway.
    return (coords[:, 1:].cpu().amax(0) + 1).tolist()


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, padding=None, bias=True, indice_key=None):
        super().__init__()
        stride_t = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
        if not (all(s == 1 for s in stride_t) and padding is None):
            raise NotImplementedError(
                "flex_gemm backend: strided SparseConv3d not supported "
                "(only submanifold conv with stride=1 and padding=None). "
                f"Got stride={stride}, padding={padding}.")
        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = ks
        self.stride = (1, 1, 1)
        self.dilation = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)

        # Initialize as the standard spconv layout (Co, Ci, Kd, Kh, Kw) so that
        # kaiming fan-in matches conv_spconv, then permute to flex_gemm's
        # (Co, Kd, Kh, Kw, Ci) expected weight layout.
        weight = torch.empty(out_channels, in_channels, *ks)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        self.weight = nn.Parameter(weight.permute(0, 2, 3, 4, 1).contiguous())

    def forward(self, x: SparseTensor) -> SparseTensor:
        # EXPLICIT_GEMM does the contraction with torch.mm/addmm (only the
        # neighbor-map build stays on Metal). The default MASKED_IMPLICIT_GEMM_SPLITK
        # Metal kernel wedges the GPU (and can panic macOS) for large K = Ci*kernel_vol,
        # e.g. Ci=2048 -> K=55296. Force the torch path until that kernel is fixed.
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        set_hashmap_ratio(flex_gemm.ops.spconv.HASHMAP_RATIO)

        Kd, Kh, Kw = self.kernel_size
        # Key by coords identity (storage ptr + N), NOT just kernel size. SparseUpsample/
        # Downsample copy the parent's _spatial_cache wholesale, so a key that ignores
        # resolution leaks a coarse-resolution neighbor map onto a finer-resolution conv
        # (e.g. N=1949 map reused at N=8088 -> OOB indices -> GPU wedge / IndexError).
        # Within a resolution every conv shares the same coords tensor (replace() reuses
        # it), so reuse still hits; across up/down the new coords give a fresh key.
        cache_key = f"flexgemm_subm_{Kw}x{Kh}x{Kd}_d{self.dilation}_p{x.coords.data_ptr()}_n{x.coords.shape[0]}"
        neighbor_cache = x.get_spatial_cache(cache_key)

        # shape is NCWHD: batch, in_channels, then the spatial extent.
        shape = torch.Size([x.shape[0], self.in_channels, *_spatial_shape(x)])
        out_feats, cache_ = sparse_submanifold_conv3d(
            x.feats, x.coords, shape, self.weight, self.bias,
            neighbor_cache, self.dilation,
        )
        if neighbor_cache is None:
            x.register_spatial_cache(cache_key, cache_)
        return x.replace(out_feats)


class SparseInverseConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flex_gemm backend: SparseInverseConv3d not supported "
            "(flex_gemm exposes submanifold conv only, no inverse/transpose conv).")

    def forward(self, x: SparseTensor) -> SparseTensor:  # pragma: no cover
        raise NotImplementedError(
            "flex_gemm backend: SparseInverseConv3d not supported.")
