import time
import math
from tqdm import tqdm
import torch
from flex_gemm.ops.grid_sample import grid_sample_3d_torch, grid_sample_3d
from utils import sphere_coords, calc_err


@torch.no_grad()
def sphere_query_pts(res, device='cuda'):
    theta, phi = torch.meshgrid(
        torch.linspace(0, 2 * math.pi, res, device=device),
        torch.linspace(-math.pi / 2, math.pi / 2, res, device=device),
        indexing='ij'
    )
    center = res / 2
    radius = res / 2 - 0.5
    pts = torch.stack([
        radius * torch.cos(theta) * torch.sin(phi) + center,
        radius * torch.sin(theta) * torch.sin(phi) + center,
        radius * torch.cos(phi) + center
    ], dim=-1)
    # pts = torch.cat([pts, torch.randn_like(pts)], dim=0)
    return pts


def benchmark_kernel(kernel_fn, *args, prepare_fn=None, num_warmup=10, num_iters=100, **kwargs):
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
    return avg_time_ms, avg_mem_gb, C


@torch.no_grad()
def grid_sample_3d_torch_nearest(feats, coords, shape, grid, **kwargs):
    return grid_sample_3d_torch(feats, coords, shape, grid.reshape(1, -1, 3), 'nearest').reshape(*grid.shape[:-1], -1)


@torch.no_grad()
def grid_sample_3d_torch_trilinear(feats, coords, shape, grid, **kwargs):
    return grid_sample_3d_torch(feats, coords, shape, grid.reshape(1, -1, 3), 'trilinear').reshape(*grid.shape[:-1], -1)


@torch.no_grad()
def grid_sample_3d_nearest(feats, coords, shape, grid, **kwargs):
    return grid_sample_3d(feats, coords, shape, grid.reshape(1, -1, 3), 'nearest').reshape(*grid.shape[:-1], -1)


@torch.no_grad()
def grid_sample_3d_trilinear(feats, coords, shape, grid, **kwargs):
    return grid_sample_3d(feats, coords, shape, grid.reshape(1, -1, 3), 'trilinear').reshape(*grid.shape[:-1], -1)


def grid_sample_3d_torch_nearest_bwd_prepare(feats, coords, shape, grid, **kwargs):
    out = grid_sample_3d_torch(feats, coords, shape, grid.reshape(1, -1, 3), 'nearest').reshape(*grid.shape[:-1], -1)
    return {
        'out': out,
        'feats': feats,
        **kwargs,
    }
    

def grid_sample_3d_torch_nearest_bwd(grad_out, out, feats):
    grad_feats = torch.autograd.grad(
        outputs=out,
        inputs=[feats],
        grad_outputs=grad_out,
        retain_graph=True,
    )[0]
    return grad_feats


def grid_sample_3d_torch_trilinear_bwd_prepare(feats, coords, shape, grid, **kwargs):
    out = grid_sample_3d_torch(feats, coords, shape, grid.reshape(1, -1, 3), 'trilinear').reshape(*grid.shape[:-1], -1)
    return {
        'out': out,
        'feats': feats,
        **kwargs,
    }
    
    
def grid_sample_3d_torch_trilinear_bwd(grad_out, out, feats):
    grad_feats = torch.autograd.grad(
        outputs=out,
        inputs=[feats],
        grad_outputs=grad_out,
        retain_graph=True,
    )[0]
    return grad_feats


def grid_sample_3d_nearest_bwd_prepare(feats, coords, shape, grid, **kwargs):
    out = grid_sample_3d(feats, coords, shape, grid.reshape(1, -1, 3), 'nearest').reshape(*grid.shape[:-1], -1)
    return {
        'out': out,
        'feats': feats,
        **kwargs,
    }
    

def grid_sample_3d_nearest_bwd(grad_out, out, feats):
    grad_feats = torch.autograd.grad(
        outputs=out,
        inputs=[feats],
        grad_outputs=grad_out,
        retain_graph=True,
    )[0]
    return grad_feats


def grid_sample_3d_trilinear_bwd_prepare(feats, coords, shape, grid, **kwargs):
    out = grid_sample_3d(feats, coords, shape, grid.reshape(1, -1, 3), 'trilinear').reshape(*grid.shape[:-1], -1)
    return {
        'out': out,
        'feats': feats,
        **kwargs,
    }
    
    
def grid_sample_3d_trilinear_bwd(grad_out, out, feats):
    grad_feats = torch.autograd.grad(
        outputs=out,
        inputs=[feats],
        grad_outputs=grad_out,
        retain_graph=True,
    )[0]
    return grad_feats


def test(kernel_functions, reference, title):
    # Matrix dimensions.
    config = [
        {'RES': 8, 'C': 2048},
        {'RES': 16, 'C': 1024},
        {'RES': 32, 'C': 512},
        {'RES': 64, 'C': 256},
        {'RES': 128, 'C': 128},
        {'RES': 256, 'C': 64},
        {'RES': 512, 'C': 32},
        {'RES': 1024, 'C': 16},
        {'RES': 2048, 'C': 8},
    ]
    
    results = {}
    for c in tqdm(config, leave=False):
        RES, C = c['RES'], c['C']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, dtype=torch.float32)
        query_pts = sphere_query_pts(RES)
        feats.requires_grad = True
        args = {
            'feats': feats,
            'coords': coords,
            'shape': shape,
            'grid': query_pts,
            'grad_out': torch.randn(*query_pts.shape[:-1], C, device=feats.device, dtype=feats.dtype),
        }

        config_key = f'RES={RES},C={C}'
        results[config_key] = {
            'time': [],
            'memory': [],
            'err_max': [],
            'err_mean': [],
        }
        
        # Benchmark the reference kernel.
        avg_time_ref, memory_ref, C_ref = benchmark_kernel(reference[0], **args, prepare_fn=reference[1])

        # Benchmark each custom kernel.
        for kernel_fn, prepare_fn in kernel_functions.values():
            avg_time, memory, C_kernel = benchmark_kernel(kernel_fn, **args, prepare_fn=prepare_fn)
            results[config_key]['time'].append(f'{avg_time:.2f} ms ({avg_time_ref/avg_time*100:.1f}%)')
            results[config_key]['memory'].append(f'{memory:.1f}G')
            if C_kernel is not None:
                err_max, err_mean = calc_err(C_kernel, C_ref)
                results[config_key]['err_max'].append(f'{err_max * 1000:.0f}‰')
                results[config_key]['err_mean'].append(f'{err_mean * 1000:.0f}‰')
            else:
                results[config_key]['err_max'].append('N/A')
                results[config_key]['err_mean'].append('N/A')
                
    # Print results as a formatted table.
    print("=" * 180)
    print(title)
    print("=" * 180)
    for m in ['time','memory', 'err_max', 'err_mean']:
        print(m.capitalize())
        print("-" * 180)
        items = [f'{"settings":<15}']
        for f in kernel_functions.keys():
            items.append(f'{f:<20}')
        print(' | '.join(items))
        print("-" * 180)
        for k, v in results.items():
            items = [f'{k:<15}']
            items.extend([f'{x:<20}' for x in v[m]])
            print(' | '.join(items))
        print("-" * 180)
        

if __name__ == "__main__":
    test(
        kernel_functions={
            'torch': (grid_sample_3d_torch_nearest, None),
            'flex_gemm': (grid_sample_3d_nearest, None),
        },
        reference=(grid_sample_3d_torch_nearest, None),
        title="Grid Sample 3D Nearest Fwd Benchmark",
    )
    
    test(
        kernel_functions={
            'torch': (grid_sample_3d_torch_trilinear, None),
            'flex_gemm': (grid_sample_3d_trilinear, None),
        },
        reference=(grid_sample_3d_torch_trilinear, None),
        title="Grid Sample 3D Trilinear Fwd Benchmark",
    )
    
    test(
        kernel_functions={
            'torch': (grid_sample_3d_torch_nearest_bwd, grid_sample_3d_torch_nearest_bwd_prepare),
            'flex_gemm': (grid_sample_3d_nearest_bwd, grid_sample_3d_nearest_bwd_prepare),
        },
        reference=(grid_sample_3d_torch_nearest_bwd, grid_sample_3d_torch_nearest_bwd_prepare),
        title="Grid Sample 3D Nearest Bwd Benchmark",
    )
    
    test(
        kernel_functions={
            'torch': (grid_sample_3d_torch_trilinear_bwd, grid_sample_3d_torch_trilinear_bwd_prepare),
            'flex_gemm': (grid_sample_3d_trilinear_bwd, grid_sample_3d_trilinear_bwd_prepare),
        },
        reference=(grid_sample_3d_torch_trilinear_bwd, grid_sample_3d_torch_trilinear_bwd_prepare),
        title="Grid Sample 3D Trilinear Bwd Benchmark",
    )
