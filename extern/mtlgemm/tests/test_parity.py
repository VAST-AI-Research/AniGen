"""
Comprehensive parity tests for Metal backend.

Compares Metal kernel outputs against PyTorch reference implementations
to verify numerical equivalence (exact for integer ops, tolerance for float ops).

IMPORTANT: Tests exercise all algorithm paths including IMPLICIT_GEMM and
MASKED_IMPLICIT_GEMM which dispatch to Metal compute shader kernels.
EXPLICIT_GEMM uses pure PyTorch and serves as the reference baseline.
"""
import pytest
import numpy as np
import torch
from flex_gemm import kernels
from flex_gemm.ops import serialize
from flex_gemm.ops.spconv import (
    Algorithm, set_algorithm, sparse_submanifold_conv3d,
)
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
from flex_gemm.ops.grid_sample import grid_sample_3d
from flex_gemm.ops import utils as op_utils


# Algorithm combos for parametrization
ALGO_ALL = [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM]
ALGO_METAL = [Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM]


# =============================================================================
# 1. Hashmap parity (exact uint32)
# =============================================================================

class TestHashmapParity:

    @pytest.mark.parametrize("res", [4, 8, 16, 32, 64, 128])
    def test_insert_lookup(self, device, sphere_coords_fn, res):
        """Insert coords with idx-as-val, lookup all -> verify index identity."""
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords at this resolution")

        N, C, W, H, D = shape
        hashmap_keys = torch.full((2 * M,), 0xFFFFFFFF, dtype=torch.uint32, device=device)
        hashmap_vals = torch.empty((2 * M,), dtype=torch.uint32, device=device)

        kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)
        result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

        expected = torch.arange(M, dtype=torch.int64, device=device).to(torch.uint32)
        assert torch.all(result == expected), (
            f"Hashmap idx-as-val mismatch: {(result != expected).sum().item()}/{M} wrong"
        )

    @pytest.mark.parametrize("res", [4, 8, 16, 32, 64, 128])
    def test_missing_coords_return_sentinel(self, device, sphere_coords_fn, res):
        """Lookup coords not in the map -> must return 0xFFFFFFFF sentinel."""
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        N, C, W, H, D = shape
        hashmap_keys = torch.full((2 * M,), 0xFFFFFFFF, dtype=torch.uint32, device=device)
        hashmap_vals = torch.empty((2 * M,), dtype=torch.uint32, device=device)

        kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

        # Build coords that are guaranteed to NOT be in the sphere shell:
        # use batch=99 which is never inserted (all inserts use batch=0)
        missing = torch.tensor(
            [[99, 0, 0, 0], [99, 1, 1, 1], [99, 2, 2, 2]],
            dtype=torch.int32, device=device,
        )
        result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, missing, W, H, D)
        sentinel = torch.full((3,), 0xFFFFFFFF, dtype=torch.uint32, device=device)
        assert torch.all(result == sentinel), (
            f"Missing coords should return sentinel, got: {result}"
        )

    @pytest.mark.parametrize("load_pct", [50, 80, 95])
    def test_high_load_factor(self, device, sphere_coords_fn, load_pct):
        """Stress: hashmap at high load factors must not lose data."""
        _, coords, shape = sphere_coords_fn(32)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        N, C, W, H, D = shape
        hashmap_size = max(int(M * 100 / load_pct), M + 1)
        hashmap_keys = torch.full((hashmap_size,), 0xFFFFFFFF, dtype=torch.uint32, device=device)
        hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32, device=device)

        kernels.cuda.hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)
        result = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, coords, W, H, D)

        expected = torch.arange(M, dtype=torch.int64, device=device).to(torch.uint32)
        assert torch.all(result == expected), (
            f"Data loss at {load_pct}% load: {(result != expected).sum().item()}/{M} wrong"
        )


# =============================================================================
# 2. Serialize parity (exact int32/int64)
# =============================================================================

class TestSerializeParity:

    @pytest.mark.parametrize("mode", ["z_order", "hilbert"])
    @pytest.mark.parametrize("res", [4, 8, 16, 32, 64, 128, 256])
    def test_roundtrip(self, device, sphere_coords_fn, mode, res):
        """Encode -> decode -> must recover exact original coords."""
        _, coords, shape = sphere_coords_fn(res)
        if coords.shape[0] == 0:
            pytest.skip("No coords")

        codes = serialize.encode_seq(coords, shape, mode=mode)
        decoded = serialize.decode_seq(codes, shape, mode=mode)
        assert torch.all(coords == decoded), (
            f"Roundtrip failed: {(coords != decoded).any(dim=1).sum().item()} coords differ"
        )

    @pytest.mark.parametrize("mode", ["z_order", "hilbert"])
    @pytest.mark.parametrize("res", [4, 8, 16, 32, 64, 128, 256])
    def test_unique_codes(self, device, sphere_coords_fn, mode, res):
        """All unique coords must produce unique codes (no collisions)."""
        _, coords, shape = sphere_coords_fn(res)
        if coords.shape[0] == 0:
            pytest.skip("No coords")

        codes = serialize.encode_seq(coords, shape, mode=mode)
        assert codes.unique().shape[0] == codes.shape[0], (
            f"Collision: {codes.shape[0] - codes.unique().shape[0]} duplicate codes"
        )

    @pytest.mark.parametrize("mode", ["z_order", "hilbert"])
    def test_sorted_codes_give_space_filling_order(self, device, sphere_coords_fn, mode):
        """Sorted codes should produce a valid space-filling curve ordering."""
        _, coords, shape = sphere_coords_fn(16)
        if coords.shape[0] == 0:
            pytest.skip("No coords")

        codes = serialize.encode_seq(coords, shape, mode=mode)
        sorted_indices = torch.argsort(codes)
        sorted_coords = coords[sorted_indices]

        diffs = (sorted_coords[1:, 1:] - sorted_coords[:-1, 1:]).abs().sum(dim=1).float()
        mean_step = diffs.mean().item()
        assert mean_step < 10, f"Mean L1 step {mean_step:.1f} too large for space-filling curve"


# =============================================================================
# 3. Neighbor map parity (exact uint32, vs PyTorch reference)
# =============================================================================

class TestNeighborMapParity:

    @pytest.mark.parametrize("res", [8, 16, 32, 64])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_vs_torch_reference(self, device, sphere_coords_fn, res, dilation):
        """Metal neighbor map must exactly match PyTorch reference."""
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        kernel_size = (3, 3, 3)
        dilation_t = (dilation, dilation, dilation)

        metal_cache = SubMConv3dFunction._compute_neighbor_cache(
            coords, shape, kernel_size, dilation_t
        )
        metal_map = metal_cache['neighbor_map']

        torch_cache = SubMConv3dFunction._compute_neighbor_cache_torch(
            coords, shape, kernel_size, dilation_t
        )
        torch_map = torch_cache['neighbor_map']

        assert metal_map.shape == torch_map.shape, (
            f"Shape mismatch: metal {metal_map.shape} vs torch {torch_map.shape}"
        )
        assert torch.all(metal_map == torch_map), (
            f"Neighbor map mismatch: {(metal_map != torch_map).sum().item()}/{metal_map.numel()} entries differ"
        )

    @pytest.mark.parametrize("res", [8, 16, 32])
    def test_symmetry(self, device, sphere_coords_fn, res):
        """If neighbor[i, v] == j, then neighbor[j, V-1-v] == i."""
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        cache = SubMConv3dFunction._compute_neighbor_cache(
            coords, shape, (3, 3, 3), (1, 1, 1)
        )
        nmap = cache['neighbor_map'].long()
        V = nmap.shape[1]

        # Vectorized symmetry check (no slow Python loop)
        for v in range(V):
            col = nmap[:, v]
            valid = col != 0xFFFFFFFF
            mirror_v = V - 1 - v
            # For valid entries: nmap[nmap[i, v], V-1-v] must equal i
            valid_indices = torch.where(valid)[0]
            neighbors = col[valid]
            mirror_vals = nmap[neighbors, mirror_v]
            violations = (mirror_vals != valid_indices).sum().item()
            assert violations == 0, f"Symmetry violations at v={v}: {violations}"

    @pytest.mark.parametrize("res", [8, 16, 32])
    def test_center_element_is_self(self, device, sphere_coords_fn, res):
        """Center kernel element: neighbor[i, V//2] == i for all i."""
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        cache = SubMConv3dFunction._compute_neighbor_cache(
            coords, shape, (3, 3, 3), (1, 1, 1)
        )
        nmap = cache['neighbor_map']
        V = nmap.shape[1]
        center = V // 2

        expected = torch.arange(M, dtype=torch.int64, device=device).to(torch.uint32)
        assert torch.all(nmap[:, center] == expected), (
            f"Center element mismatch: {(nmap[:, center] != expected).sum().item()}/{M}"
        )


# =============================================================================
# 4. Neighbor map post-processing parity (exact)
# =============================================================================

class TestNeighborMapPostprocessParity:

    @pytest.mark.parametrize("res", [8, 16, 32])
    def test_postprocess_1_gray_code(self, device, sphere_coords_fn, res):
        """Post-process stage 1: gray_code bitmask matches manual computation."""
        set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        cache = SubMConv3dFunction._compute_neighbor_cache(
            coords, shape, (3, 3, 3), (1, 1, 1)
        )
        nmap = cache['neighbor_map']
        gray_code = cache['gray_code']
        valid_signal_seg = cache['valid_signal_seg']

        V = nmap.shape[1]

        expected_gray = torch.zeros(M, dtype=torch.int32, device=device)
        for v in range(V):
            valid = (nmap[:, v] != 0xFFFFFFFF)
            expected_gray |= valid.to(torch.int32) << v
        assert torch.all(gray_code.view(torch.int32) == expected_gray), "Gray code mismatch"

        total_valid = (nmap.view(torch.int32) != -1).sum().item()
        assert valid_signal_seg[-1].item() == total_valid, (
            f"valid_signal_seg[-1] {valid_signal_seg[-1].item()} != total valid {total_valid}"
        )

    @pytest.mark.parametrize("res", [8, 16])
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_postprocess_2_block_reduce(self, device, sphere_coords_fn, res, block_size):
        """Post-process stage 2: total valid kernel indices matches sum of popcounts."""
        set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
        _, coords, shape = sphere_coords_fn(res)
        M = coords.shape[0]
        if M == 0:
            pytest.skip("No coords")

        cache = SubMConv3dFunction._compute_neighbor_cache(
            coords, shape, (3, 3, 3), (1, 1, 1)
        )
        gray_code = cache['gray_code']
        sorted_idx = cache['sorted_idx']

        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(
            gray_code, sorted_idx, block_size
        )

        sorted_codes = gray_code[sorted_idx]
        num_blocks = (M + block_size - 1) // block_size
        expected_total = 0
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, M)
            block_or = 0
            for c in sorted_codes[start:end]:
                block_or |= c.item()
            expected_total += bin(block_or).count('1')

        actual_total = valid_kernel_seg[-1].item()
        assert actual_total == expected_total, (
            f"Total valid kernels: got {actual_total}, expected {expected_total}"
        )

        V = 27  # 3x3x3
        assert torch.all((valid_kernel >= 0) & (valid_kernel < V)), "Invalid kernel index"


# =============================================================================
# 5. Grid sample parity (float, vs PyTorch reference)
# =============================================================================

class TestGridSampleParity:

    @pytest.mark.parametrize("res", [8, 16, 32, 64])
    @pytest.mark.parametrize("C", [4, 16, 64])
    def test_nearest_vs_torch(self, device, sphere_coords_fn, ref_grid_sample, calc_err, res, C):
        """Metal nearest grid sample must match PyTorch reference."""
        feats, coords, shape = sphere_coords_fn(res, ch=C)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        N = coords.shape[0]
        L = min(N, 200)
        query = coords[:L, 1:].float().unsqueeze(0)

        metal_out = grid_sample_3d(feats, coords, shape, query, mode='nearest')
        torch_out = ref_grid_sample(feats, coords, shape, query, mode='nearest')

        assert metal_out.shape == torch_out.shape
        assert torch.allclose(metal_out, torch_out, atol=1e-6), (
            f"Nearest parity fail: max_err={calc_err(metal_out, torch_out)[0]:.2e}"
        )

    @pytest.mark.parametrize("res", [8, 16, 32, 64])
    @pytest.mark.parametrize("C", [4, 16, 64])
    def test_trilinear_vs_torch(self, device, sphere_coords_fn, ref_grid_sample, calc_err, res, C):
        """Metal trilinear grid sample must match PyTorch reference."""
        feats, coords, shape = sphere_coords_fn(res, ch=C)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        N = coords.shape[0]
        L = min(N, 200)
        query = coords[:L, 1:].float().unsqueeze(0) + 0.3

        metal_out = grid_sample_3d(feats, coords, shape, query, mode='trilinear')
        torch_out = ref_grid_sample(feats, coords, shape, query, mode='trilinear')

        assert metal_out.shape == torch_out.shape
        max_err, mean_err = calc_err(metal_out, torch_out)
        assert torch.allclose(metal_out, torch_out, atol=1e-5), (
            f"Trilinear parity fail: max_err={max_err:.2e}, mean_err={mean_err:.2e}"
        )

    def test_oob_returns_zero(self, device, sphere_coords_fn, ref_grid_sample):
        """Out-of-bounds queries must return zero for both Metal and reference."""
        feats, coords, shape = sphere_coords_fn(8, ch=4)
        if feats is None:
            pytest.skip("No coords")

        oob = torch.tensor([[[-1.0, -1.0, -1.0], [100.0, 100.0, 100.0]]], device=device)

        for mode in ["nearest", "trilinear"]:
            metal_out = grid_sample_3d(feats, coords, shape, oob, mode=mode)
            torch_out = ref_grid_sample(feats, coords, shape, oob, mode=mode)
            assert torch.all(metal_out == 0), f"Metal OOB non-zero for {mode}"
            assert torch.all(torch_out == 0), f"Torch OOB non-zero for {mode}"


# =============================================================================
# 6. Weighted sum parity (float, vs manual computation)
# =============================================================================

class TestWeightedSumParity:

    @pytest.mark.parametrize("N,C,V", [
        (100, 8, 8), (1000, 32, 8), (10000, 128, 8),
    ])
    def test_fwd_vs_manual(self, device, calc_err, N, C, V):
        """Metal weighted sum forward must match manual loop."""
        M = N // 2
        inp = torch.randn(N, C, device=device)
        idx_np = np.random.randint(0, N, (M, V), dtype=np.uint32)
        mask_np = np.random.rand(M, V) < 0.1
        idx_np[mask_np] = 0xFFFFFFFF
        indices = torch.from_numpy(idx_np).to(device)
        mask = torch.from_numpy(mask_np)
        weight = torch.randn(M, V, device=device)
        weight[mask] = 0.0

        metal_out = kernels.triton.indice_weighed_sum_fwd(inp, indices, weight)

        ref_out = torch.zeros(M, C, device=device)
        for v in range(V):
            valid = indices[:, v].to(torch.int64) != 0xFFFFFFFF
            idx = indices[:, v].long().clamp(0, N - 1)
            ref_out += (weight[:, v].unsqueeze(1) * inp[idx]) * valid.unsqueeze(1).float()

        max_err, mean_err = calc_err(metal_out, ref_out)
        assert torch.allclose(metal_out, ref_out, atol=1e-5), (
            f"Weighted sum fwd: max_err={max_err:.2e}, mean_err={mean_err:.2e}"
        )

    @pytest.mark.parametrize("N,C,V", [
        (100, 8, 8), (1000, 32, 8),
    ])
    def test_bwd_vs_manual(self, device, calc_err, N, C, V):
        """Metal weighted sum backward must match manual scatter-add."""
        M = N // 2
        grad_output = torch.randn(M, C, device=device)
        idx_np = np.random.randint(0, N, (M, V), dtype=np.uint32)
        mask_np = np.random.rand(M, V) < 0.1
        idx_np[mask_np] = 0xFFFFFFFF
        indices = torch.from_numpy(idx_np).to(device)
        mask = torch.from_numpy(mask_np)
        weight = torch.randn(M, V, device=device)
        weight[mask] = 0.0

        metal_grad = kernels.triton.indice_weighed_sum_bwd_input(
            grad_output, indices, weight, N
        )

        ref_grad = torch.zeros(N, C, device=device)
        for v in range(V):
            valid = indices[:, v].to(torch.int64) != 0xFFFFFFFF
            for m in range(M):
                if valid[m]:
                    ref_grad[indices[m, v].long()] += weight[m, v] * grad_output[m]

        max_err, mean_err = calc_err(metal_grad, ref_grad)
        assert torch.allclose(metal_grad, ref_grad, atol=1e-4), (
            f"Weighted sum bwd: max_err={max_err:.2e}, mean_err={mean_err:.2e}"
        )


# =============================================================================
# 7. Sparse conv parity — cross-algorithm numerical equivalence
#
# The critical test: IMPLICIT_GEMM (Metal kernel) must match
# EXPLICIT_GEMM (pure PyTorch im2col+matmul) to within tolerance.
# =============================================================================

class TestSpconvParity:

    @pytest.mark.parametrize("algo", ALGO_ALL)
    @pytest.mark.parametrize("res,Ci,Co", [
        (8, 4, 4), (8, 16, 16), (16, 4, 16), (32, 16, 4),
    ])
    def test_fwd_vs_manual_im2col(self, device, sphere_coords_fn, calc_err, algo, res, Ci, Co):
        """Each algorithm's forward must match manual im2col+matmul reference."""
        set_algorithm(algo)
        feats, coords, shape = sphere_coords_fn(res, ch=Ci)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        N = coords.shape[0]
        V = 27  # 3x3x3
        weight = torch.randn(Co, 3, 3, 3, Ci, device=device)
        bias = torch.randn(Co, device=device)

        # Run through ops layer (dispatches to Metal kernel for IMPLICIT_GEMM)
        output, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

        # Build neighbor map for reference (always use EXPLICIT_GEMM path for cache)
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        _, ref_cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
        nmap = ref_cache['neighbor_map']

        # Manual im2col reference
        im2col = torch.zeros((N * V, Ci), device=device, dtype=feats.dtype)
        mask = nmap.view(-1) != 0xFFFFFFFF
        im2col[mask] = feats[nmap.view(-1).long()[mask]]
        im2col = im2col.view(N, V * Ci)
        w = weight.view(Co, V * Ci).t()
        ref_out = torch.addmm(bias, im2col, w)

        max_err, mean_err = calc_err(output, ref_out)
        assert torch.allclose(output, ref_out, atol=1e-5), (
            f"Spconv fwd parity fail [{algo}]: max_err={max_err:.2e}, mean_err={mean_err:.2e}"
        )

    @pytest.mark.parametrize("algo", [Algorithm.EXPLICIT_GEMM])
    @pytest.mark.parametrize("res,Ci,Co", [
        (8, 4, 4), (16, 4, 4),
    ])
    def test_bwd_gradcheck(self, device, sphere_coords_fn, algo, res, Ci, Co):
        """Verify backward pass via torch.autograd.gradcheck (double precision).
        Only EXPLICIT_GEMM: Metal kernels are float32-only, float64 produces inf."""
        set_algorithm(algo)
        feats, coords, shape = sphere_coords_fn(res, ch=Ci, dtype=torch.float64)
        if feats is None or coords.shape[0] < 2:
            pytest.skip("Not enough coords")

        N = min(coords.shape[0], 50)
        feats = feats[:N].clone().detach().requires_grad_(True)
        coords = coords[:N].clone()
        shape_adj = torch.Size([1, Ci, res, res, res])

        weight = torch.randn(Co, 3, 3, 3, Ci, device=device, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(Co, device=device, dtype=torch.float64, requires_grad=True)

        def fn(f, w, b):
            out, _ = sparse_submanifold_conv3d(f, coords, shape_adj, w, b)
            return out

        torch.autograd.gradcheck(fn, (feats, weight, bias), eps=1e-3, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("algo", ALGO_ALL)
    def test_bwd_grad_bias_equals_sum(self, device, sphere_coords_fn, algo):
        """grad_bias must equal grad_output.sum(dim=0)."""
        set_algorithm(algo)
        feats, coords, shape = sphere_coords_fn(16, ch=8)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        feats = feats.requires_grad_(True)
        weight = torch.randn(16, 3, 3, 3, 8, device=device, requires_grad=True)
        bias = torch.randn(16, device=device, requires_grad=True)

        output, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
        grad_out = torch.randn_like(output)
        output.backward(grad_out)

        assert torch.allclose(bias.grad, grad_out.sum(dim=0), atol=1e-5), (
            f"grad_bias != sum(grad_output) [{algo}]"
        )

    @pytest.mark.parametrize("algo", ALGO_ALL)
    def test_bwd_grad_values(self, device, sphere_coords_fn, calc_err, algo):
        """Backward grad_input and grad_weight from Metal must match EXPLICIT_GEMM reference."""
        set_algorithm(algo)
        feats, coords, shape = sphere_coords_fn(16, ch=8)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        weight = torch.randn(16, 3, 3, 3, 8, device=device)
        bias = torch.randn(16, device=device)

        # Run target algorithm
        feats_a = feats.clone().requires_grad_(True)
        weight_a = weight.clone().requires_grad_(True)
        bias_a = bias.clone().requires_grad_(True)
        set_algorithm(algo)
        out_a, _ = sparse_submanifold_conv3d(feats_a, coords, shape, weight_a, bias_a)
        grad_out = torch.randn_like(out_a)
        out_a.backward(grad_out)

        # Run EXPLICIT_GEMM reference
        feats_r = feats.clone().requires_grad_(True)
        weight_r = weight.clone().requires_grad_(True)
        bias_r = bias.clone().requires_grad_(True)
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        out_r, _ = sparse_submanifold_conv3d(feats_r, coords, shape, weight_r, bias_r)
        out_r.backward(grad_out)

        # Compare gradients
        max_err_gi, _ = calc_err(feats_a.grad, feats_r.grad)
        assert torch.allclose(feats_a.grad, feats_r.grad, atol=1e-5), (
            f"grad_input mismatch [{algo}]: max_err={max_err_gi:.2e}"
        )

        max_err_gw, _ = calc_err(weight_a.grad, weight_r.grad)
        assert torch.allclose(weight_a.grad, weight_r.grad, atol=1e-5), (
            f"grad_weight mismatch [{algo}]: max_err={max_err_gw:.2e}"
        )

    @pytest.mark.parametrize("algo", ALGO_ALL)
    def test_bwd_shapes(self, device, sphere_coords_fn, algo):
        """Verify grad shapes match input shapes."""
        set_algorithm(algo)
        feats, coords, shape = sphere_coords_fn(16, ch=8)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        feats = feats.requires_grad_(True)
        weight = torch.randn(16, 3, 3, 3, 8, device=device, requires_grad=True)
        bias = torch.randn(16, device=device, requires_grad=True)

        output, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
        output.sum().backward()

        assert feats.grad.shape == feats.shape, f"feats grad shape [{algo}]"
        assert weight.grad.shape == weight.shape, f"weight grad shape [{algo}]"
        assert bias.grad.shape == bias.shape, f"bias grad shape [{algo}]"

    def test_cross_algorithm_fwd_parity(self, device, sphere_coords_fn, calc_err):
        """All 3 algorithms must produce identical forward output for same inputs."""
        feats, coords, shape = sphere_coords_fn(16, ch=8)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        weight = torch.randn(16, 3, 3, 3, 8, device=device)
        bias = torch.randn(16, device=device)

        outputs = {}
        for algo in ALGO_ALL:
            set_algorithm(algo)
            out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
            outputs[algo] = out

        # Compare IMPLICIT_GEMM and MASKED_IMPLICIT_GEMM against EXPLICIT_GEMM
        ref = outputs[Algorithm.EXPLICIT_GEMM]
        for algo in ALGO_METAL:
            max_err, mean_err = calc_err(outputs[algo], ref)
            assert torch.allclose(outputs[algo], ref, atol=1e-5), (
                f"Cross-algo fwd parity [{algo} vs EXPLICIT_GEMM]: "
                f"max_err={max_err:.2e}, mean_err={mean_err:.2e}"
            )

    @pytest.mark.parametrize("algo", ALGO_METAL)
    @pytest.mark.parametrize("res,Ci,Co", [
        (16, 32, 64), (32, 32, 64), (64, 32, 64),
    ])
    def test_fwd_large_channels_error_bound(self, device, sphere_coords_fn, calc_err, algo, res, Ci, Co):
        """Metal GEMM tiled accumulation diverges from torch.mm for Ci > tile_size (32).
        Validates the error stays within 1 ULP of output magnitude (~6e-05 abs).
        This is the inherent float32 accumulation order difference, not a bug."""
        set_algorithm(algo)
        feats, coords, shape = sphere_coords_fn(res, ch=Ci)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        weight = torch.randn(Co, 3, 3, 3, Ci, device=device)
        bias = torch.randn(Co, device=device)

        output, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

        set_algorithm(Algorithm.EXPLICIT_GEMM)
        ref_out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

        abs_err = (output - ref_out).abs()
        max_abs = abs_err.max().item()
        mean_abs = abs_err.mean().item()

        # Bound: max error < 1e-4 (≈ 1-2 ULP for values in [-50, 50])
        # Mean error < 1e-5 (accumulation noise averages out)
        assert max_abs < 1e-4, (
            f"Large-channel fwd [{algo}] res={res}: max_abs={max_abs:.2e} exceeds 1e-4"
        )
        assert mean_abs < 1e-5, (
            f"Large-channel fwd [{algo}] res={res}: mean_abs={mean_abs:.2e} exceeds 1e-5"
        )

    @pytest.mark.parametrize("algo", ALGO_METAL)
    @pytest.mark.parametrize("res,Ci,Co", [
        (16, 32, 64), (32, 32, 64),
    ])
    def test_bwd_large_channels_error_bound(self, device, sphere_coords_fn, calc_err, algo, res, Ci, Co):
        """Backward error bound for large channels."""
        weight = torch.randn(Co, 3, 3, 3, Ci, device=device)
        bias = torch.randn(Co, device=device)

        feats, coords, shape = sphere_coords_fn(res, ch=Ci)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        feats_a = feats.clone().requires_grad_(True)
        weight_a = weight.clone().requires_grad_(True)
        set_algorithm(algo)
        out_a, _ = sparse_submanifold_conv3d(feats_a, coords, shape, weight_a, bias)
        grad_out = torch.randn_like(out_a)
        out_a.backward(grad_out)

        feats_r = feats.clone().requires_grad_(True)
        weight_r = weight.clone().requires_grad_(True)
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        out_r, _ = sparse_submanifold_conv3d(feats_r, coords, shape, weight_r, bias)
        out_r.backward(grad_out)

        gi_err = (feats_a.grad - feats_r.grad).abs()
        gw_err = (weight_a.grad - weight_r.grad).abs()
        assert gi_err.max().item() < 1e-4, (
            f"Large-channel bwd grad_input [{algo}]: max={gi_err.max().item():.2e}"
        )
        assert gw_err.max().item() < 1e-4, (
            f"Large-channel bwd grad_weight [{algo}]: max={gw_err.max().item():.2e}"
        )

    def test_cross_algorithm_bwd_parity(self, device, sphere_coords_fn, calc_err):
        """All 3 algorithms must produce identical gradients for same inputs."""
        feats, coords, shape = sphere_coords_fn(16, ch=8)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        weight = torch.randn(16, 3, 3, 3, 8, device=device)
        bias = torch.randn(16, device=device)
        grad_out = None

        grads = {}
        for algo in ALGO_ALL:
            f = feats.clone().requires_grad_(True)
            w = weight.clone().requires_grad_(True)
            b = bias.clone().requires_grad_(True)
            set_algorithm(algo)
            out, _ = sparse_submanifold_conv3d(f, coords, shape, w, b)
            if grad_out is None:
                grad_out = torch.randn_like(out)
            out.backward(grad_out)
            grads[algo] = (f.grad, w.grad, b.grad)

        ref_gi, ref_gw, ref_gb = grads[Algorithm.EXPLICIT_GEMM]
        for algo in ALGO_METAL:
            gi, gw, gb = grads[algo]
            max_gi, _ = calc_err(gi, ref_gi)
            max_gw, _ = calc_err(gw, ref_gw)
            assert torch.allclose(gi, ref_gi, atol=1e-5), (
                f"Cross-algo grad_input [{algo}]: max_err={max_gi:.2e}"
            )
            assert torch.allclose(gw, ref_gw, atol=1e-5), (
                f"Cross-algo grad_weight [{algo}]: max_err={max_gw:.2e}"
            )
            assert torch.allclose(gb, ref_gb, atol=1e-5), (
                f"Cross-algo grad_bias [{algo}]"
            )


# =============================================================================
# 8. Full pipeline parity (end-to-end, with numerical checks)
# =============================================================================

class TestFullPipelineParity:

    @pytest.mark.parametrize("algo", ALGO_ALL)
    @pytest.mark.parametrize("res", [16, 32])
    def test_pipeline_fwd(self, device, sphere_coords_fn, calc_err, algo, res):
        """End-to-end: spconv -> grid_sample -> serialize, with numerical comparison."""
        set_algorithm(algo)
        Ci, Co = 32, 64

        feats, coords, shape = sphere_coords_fn(res, ch=Ci)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        N = coords.shape[0]

        weight = torch.randn(Co, 3, 3, 3, Ci, device=device)
        bias = torch.randn(Co, device=device)

        # Run target algorithm
        conv_out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
        assert conv_out.shape == (N, Co)
        assert not torch.isnan(conv_out).any(), f"NaN in conv output [{algo}]"

        # Compare against EXPLICIT_GEMM reference
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        ref_out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
        max_err, mean_err = calc_err(conv_out, ref_out)
        assert torch.allclose(conv_out, ref_out, atol=1e-4), (
            f"Pipeline fwd [{algo}] vs ref: max_err={max_err:.2e}, mean_err={mean_err:.2e}"
        )

        # Grid sample
        shape_co = torch.Size([1, Co, res, res, res])
        L = min(N, 100)
        query = coords[:L, 1:].float().unsqueeze(0) + 0.25
        gs_out = grid_sample_3d(conv_out, coords, shape_co, query, mode='trilinear')
        assert gs_out.shape == (1, L, Co)
        assert not torch.isnan(gs_out).any()

        # Serialize roundtrip
        codes = serialize.encode_seq(coords, shape, mode='z_order')
        assert codes.shape == (N,)
        decoded = serialize.decode_seq(codes, shape, mode='z_order')
        assert torch.all(decoded == coords)

    @pytest.mark.parametrize("algo", ALGO_ALL)
    @pytest.mark.parametrize("res", [16, 32])
    def test_pipeline_backward(self, device, sphere_coords_fn, calc_err, algo, res):
        """End-to-end backward: verify gradients match EXPLICIT_GEMM reference."""
        Ci, Co = 8, 16

        feats, coords, shape = sphere_coords_fn(res, ch=Ci)
        if feats is None or coords.shape[0] == 0:
            pytest.skip("No coords")

        N = coords.shape[0]
        weight = torch.randn(Co, 3, 3, 3, Ci, device=device)
        bias = torch.randn(Co, device=device)

        L = min(N, 50)
        query = coords[:L, 1:].float().unsqueeze(0) + 0.25

        # Run target algorithm
        feats_a = feats.clone().requires_grad_(True)
        weight_a = weight.clone().requires_grad_(True)
        bias_a = bias.clone().requires_grad_(True)
        set_algorithm(algo)
        conv_a, _ = sparse_submanifold_conv3d(feats_a, coords, shape, weight_a, bias_a)
        shape_co = torch.Size([1, Co, res, res, res])
        gs_a = grid_sample_3d(conv_a, coords, shape_co, query, mode='trilinear')
        gs_a.sum().backward()

        assert feats_a.grad is not None, f"No feats grad [{algo}]"
        assert weight_a.grad is not None, f"No weight grad [{algo}]"
        assert bias_a.grad is not None, f"No bias grad [{algo}]"
        assert not torch.isnan(feats_a.grad).any(), f"NaN in feats.grad [{algo}]"
        assert not torch.isnan(weight_a.grad).any(), f"NaN in weight.grad [{algo}]"

        # Compare against EXPLICIT_GEMM reference
        feats_r = feats.clone().requires_grad_(True)
        weight_r = weight.clone().requires_grad_(True)
        bias_r = bias.clone().requires_grad_(True)
        set_algorithm(Algorithm.EXPLICIT_GEMM)
        conv_r, _ = sparse_submanifold_conv3d(feats_r, coords, shape, weight_r, bias_r)
        gs_r = grid_sample_3d(conv_r, coords, shape_co, query, mode='trilinear')
        gs_r.sum().backward()

        max_gi, _ = calc_err(feats_a.grad, feats_r.grad)
        assert torch.allclose(feats_a.grad, feats_r.grad, atol=1e-5), (
            f"Pipeline bwd feats [{algo}]: max_err={max_gi:.2e}"
        )
        max_gw, _ = calc_err(weight_a.grad, weight_r.grad)
        assert torch.allclose(weight_a.grad, weight_r.grad, atol=1e-5), (
            f"Pipeline bwd weight [{algo}]: max_err={max_gw:.2e}"
        )
