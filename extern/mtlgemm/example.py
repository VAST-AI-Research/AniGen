import torch
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from tests.spconv_fwd import sphere_coords

# Sparse voxel shell
feats, coords, shape = sphere_coords(64, 256, dtype=torch.float16, device='cuda')

# Weight and bias
Ci, Co = 256, 256
Ks = 3
weight = torch.randn(Co, Ks, Ks, Ks, Ci, dtype=torch.float16, device='cuda', requires_grad=True)
bias = torch.randn(Co, dtype=torch.float16, device='cuda', requires_grad=True)

# Set algorithm: Masked + Split-K
flex_gemm.ops.spconv.set_algorithm(
    flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK
)

out_feats, neighbor_cache = sparse_submanifold_conv3d(
    feats, coords, shape,
    weight, bias,
)

out_feats.sum().backward()
