import time
from tqdm import tqdm
import torch
import flex_gemm
from flex_gemm.ops.spconv import SubMConv3dFunction
from utils import sphere_coords


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


def egemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    
    
def igemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def igemmk_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def migemm_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }
    

def migemmk_prepare_fn(coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    return {
        'weight': weight,
        'neighbor_cache': neighbor_cache,
        **kwargs,
    }


def test_neighbor_cache():
    # Matrix dimensions.
    config = [
        {'RES': 8, 'C': 1024},
        {'RES': 16, 'C': 1024},
        {'RES': 32, 'C': 1024},
        {'RES': 64, 'C': 1024},
        {'RES': 128, 'C': 512},
        {'RES': 256, 'C': 256},
        {'RES': 512, 'C': 128},
        {'RES': 1024, 'C': 64},
        {'RES': 2048, 'C': 32},
    ]
    
    # List of custom kernel functions.
    kernel_functions = {
        'egemm': (egemm_prepare_fn, None),
        'igemm': (igemm_prepare_fn, None),
        'igemmk': (igemmk_prepare_fn, None),
        'migemm': (migemm_prepare_fn, None),
        'migemmk': (migemmk_prepare_fn, None),
    }
    
    results = {}
    for c in tqdm(config, leave=False):
        RES, C = c['RES'], c['C']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, dtype=torch.float16)
        weight = torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype)
        args = {
            'coords': coords,
            'shape': shape,
            'weight': weight,
        }

        config_key = f'RES={RES},C={C}'
        results[config_key] = []

        # Benchmark each custom kernel.
        for kernel_fn, prepare_fn in kernel_functions.values():
            avg_time, memory, C_kernel = benchmark_kernel(kernel_fn, **args, prepare_fn=prepare_fn)
            results[config_key].append(f'{avg_time:.3f}/{memory:.3f}G')

    # Print results as a formatted table.
    print("\nNeighbor Cache Benchmark Results")
    print("-" * 180)
    items = [f'{"settings":<15}']
    for f in kernel_functions.keys():
        items.append(f'{f:<20}')
    print(' | '.join(items))
    print("-" * 180)
    for k, v in results.items():
        items = [f'{k:<15}']
        items.extend([f'{x:<20}' for x in v])
        print(' | '.join(items))
        

if __name__ == "__main__":
    test_neighbor_cache()
