"""u64 hashmap on MPS — parity vs the u32 hashmap at shapes that fit in u32.

Before this commit the u64 MPS path was gated off with a hard TORCH_CHECK
telling users to "use a smaller spatial volume that fits in uint32 keys".
That closed off spatial grids > 2^32 entirely on MPS. The packed u64 variant
(interleaved [lo, hi] u32 pairs viewed zero-copy) unblocks that.

Testing at u32-fittable shapes so we can compare u64 vs u32 results
directly. The algorithm is identical between the two; the difference is
only in how the key is stored/hashed.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import pytest
import torch

import flex_gemm


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="u64 MPS hashmap test requires MPS.",
)


def _build_hashmap(N, dtype, device):
    sentinel_keys = torch.full((N,), -1 if dtype == torch.int64 else 0xFFFFFFFF,
                               dtype=dtype).to(device)
    vals = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    return sentinel_keys, vals


@needs_mps
@pytest.mark.parametrize("N,M,W,H,D", [
    (256, 64, 8, 8, 8),
    (1024, 256, 16, 16, 16),
    (8192, 2048, 32, 32, 32),
])
def test_u64_vs_u32_insert_lookup_mps(N, M, W, H, D):
    """At volumes that fit in u32, u64 and u32 hashmaps should yield identical
    lookup results — same hash (murmur3_64 vs murmur3_32 pre-masking), same
    linear probing, same slot assignments modulo hash function. We don't
    require slot equality, only value equality."""
    torch.manual_seed(0xB00B ^ (N * 31) ^ (W * 17) ^ (H * 13) ^ (D * 7))
    device = "mps"

    # Random unique coords.
    flat = torch.randperm(W * H * D)[:M]
    cz = flat // (W * H)
    cy = (flat // W) % H
    cx = flat % W
    coords = torch.stack([torch.zeros(M, dtype=torch.int32), cx.int(), cy.int(), cz.int()], dim=-1).to(device)
    values = torch.arange(M, dtype=torch.int64).to(torch.uint32).to(device)

    # u32 path
    hk32 = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    hv32 = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    flex_gemm.kernels.cuda.hashmap_insert_3d_cuda(hk32, hv32, coords, values, W, H, D)
    out32 = flex_gemm.kernels.cuda.hashmap_lookup_3d_cuda(hk32, hv32, coords, W, H, D)

    # u64 path — same inputs, different key dtype. Sentinel is UINT64_MAX,
    # which packs to 0xFFFFFFFF in both hi and lo. torch.full with uint64 on
    # MPS is spotty across builds; construct on CPU then transfer.
    hk64_cpu = torch.empty(N, dtype=torch.uint64)
    hk64_cpu.fill_(0xFFFFFFFFFFFFFFFF)
    hk64 = hk64_cpu.to(device)
    hv64 = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    flex_gemm.kernels.cuda.hashmap_insert_3d_cuda(hk64, hv64, coords, values, W, H, D)
    out64 = flex_gemm.kernels.cuda.hashmap_lookup_3d_cuda(hk64, hv64, coords, W, H, D)

    assert out32.device.type == "mps"
    assert out64.device.type == "mps"

    out32_cpu = out32.cpu()
    out64_cpu = out64.cpu()
    values_cpu = values.cpu()

    # Every inserted coord must look up to its original value in both paths.
    assert torch.equal(out32_cpu, values_cpu), f"u32 insert/lookup self-consistency failed"
    assert torch.equal(out64_cpu, values_cpu), f"u64 insert/lookup self-consistency failed"
    # And u64 must match u32 — same coords → same values.
    assert torch.equal(out32_cpu, out64_cpu), f"u64 vs u32 MPS parity failed"


@needs_mps
def test_u64_idx_as_val_mps():
    """Tests the idx-as-val insert (auto-assigns values as tid) — used by
    the submanifold_conv neighbor_map builder. Parity vs u32."""
    torch.manual_seed(0xB00C)
    device = "mps"
    N, M, W, H, D = 1024, 256, 16, 16, 16

    flat = torch.randperm(W * H * D)[:M]
    cz = flat // (W * H)
    cy = (flat // W) % H
    cx = flat % W
    coords = torch.stack([torch.zeros(M, dtype=torch.int32), cx.int(), cy.int(), cz.int()], dim=-1).to(device)

    hk32 = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    hv32 = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    flex_gemm.kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hk32, hv32, coords, W, H, D)
    out32 = flex_gemm.kernels.cuda.hashmap_lookup_3d_cuda(hk32, hv32, coords, W, H, D)

    hk64_cpu = torch.empty(N, dtype=torch.uint64)
    hk64_cpu.fill_(0xFFFFFFFFFFFFFFFF)
    hk64 = hk64_cpu.to(device)
    hv64 = torch.full((N,), 0xFFFFFFFF, dtype=torch.uint32).to(device)
    flex_gemm.kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hk64, hv64, coords, W, H, D)
    out64 = flex_gemm.kernels.cuda.hashmap_lookup_3d_cuda(hk64, hv64, coords, W, H, D)

    # Each lookup must return tid in [0, M) and be a valid permutation.
    out32_cpu, out64_cpu = out32.cpu(), out64.cpu()
    assert out32_cpu.sort().values.equal(torch.arange(M, dtype=torch.int64).to(torch.uint32)), \
        "u32 idx_as_val did not produce a permutation of [0, M)"
    assert out64_cpu.sort().values.equal(torch.arange(M, dtype=torch.int64).to(torch.uint32)), \
        "u64 idx_as_val did not produce a permutation of [0, M)"
    # And they must be identical.
    assert torch.equal(out32_cpu, out64_cpu), "u64 vs u32 idx_as_val MPS parity failed"
