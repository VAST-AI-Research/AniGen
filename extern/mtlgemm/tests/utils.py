import time
import torch


@torch.no_grad()
def sphere_coords(res, ch, device='cuda', dtype=torch.float):
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
    feats = torch.randn(coords.shape[0], ch, device=device, dtype=dtype)
    return feats.contiguous(), coords.contiguous(), torch.Size([1, ch, res, res, res])


def calc_err(src, ref):
    abs_err = (src - ref).float().abs()
    rel_err = abs_err / torch.clamp_min(ref.float().abs(), 1e-6)
    err = torch.minimum(abs_err, rel_err)
    return err.max().item(), err.mean().item()


def benchmark_kernel(kernel_fn, *args, prepare_fn=None, num_warmup=2, num_iters=20, **kwargs):
    try:
        if prepare_fn is not None:
            kwargs = prepare_fn(*args, **kwargs)
            args = tuple()
        # Warmup iterations.
        for _ in range(num_warmup):
            C = kernel_fn(*args, **kwargs)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        # Timing iterations.
        start = time.time()
        for _ in range(num_iters):
            C = kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        memory = torch.cuda.max_memory_allocated() / 1024**3
        avg_time_ms = (elapsed / num_iters) * 1000.0
        avg_mem_gb = memory
        if isinstance(C, tuple):
            C = torch.cat([c.detach().flatten() for c in C if c is not None], dim=0)
    except Exception as e:
        if isinstance(e, RuntimeError) and 'out of memory' in str(e):
            print('WARNING: OOM error occurred during benchmarking. Skipping this run.')
            return None, 'OOM', None
        else:
            raise e
    return avg_time_ms, avg_mem_gb, C


def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()