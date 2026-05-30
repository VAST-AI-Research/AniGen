import pytest
import platform
import torch


def get_device():
    # Metal backend uses StorageModeShared buffers — tensors are CPU,
    # GPU dispatch is handled internally by the Metal extension.
    # No MPS or CUDA needed.
    if platform.system() == "Darwin":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def device():
    return get_device()


@pytest.fixture
def sphere_coords_fn(device):
    """Returns a function that generates sphere shell coordinates at given resolution."""
    def _sphere_coords(res, ch=0, dtype=torch.float):
        l_coords = []
        for i in range(0, res, 256):
            for j in range(0, res, 256):
                for k in range(0, res, 256):
                    coords = torch.stack(torch.meshgrid(
                        torch.arange(i, min(i + 256, res), device=device),
                        torch.arange(j, min(j + 256, res), device=device),
                        torch.arange(k, min(k + 256, res), device=device),
                        indexing='ij'
                    ), dim=-1).int().contiguous()
                    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
                    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
                    coords = torch.nonzero(active).int() + torch.tensor([i, j, k], device=device, dtype=torch.int32)
                    l_coords.append(coords)
        coords = torch.cat(l_coords, dim=0)
        coords = torch.cat([torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32), coords], dim=-1)
        feats = torch.randn(coords.shape[0], ch, device=device, dtype=dtype) if ch > 0 else None
        return feats, coords, torch.Size([1, ch, res, res, res])
    return _sphere_coords


@pytest.fixture
def calc_err():
    """Returns calc_err(src, ref) -> (max_err, mean_err) using combined abs/rel metric."""
    def _calc_err(src, ref):
        abs_err = (src - ref).float().abs()
        rel_err = abs_err / torch.clamp_min(ref.float().abs(), 1e-6)
        err = torch.minimum(abs_err, rel_err)
        return err.max().item(), err.mean().item()
    return _calc_err


@pytest.fixture
def ref_grid_sample():
    """Returns the PyTorch reference grid_sample_3d_torch function."""
    from flex_gemm.ops.grid_sample import grid_sample_3d_torch
    return grid_sample_3d_torch
