from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import flex_gemm
from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d
from utils import sphere_coords, benchmark_kernel, zero_grad


DTYPE = torch.float16
allow_tf32 = True


def egemm_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = allow_tf32

    # Create random weight and bias matrices.
    params = torch.nn.ParameterDict()
    for i in range(L):
        weight = torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype))
        bias = torch.nn.Parameter(torch.randn(C, device=feats.device, dtype=feats.dtype))
        params[f'layer{i}'] = {'weight': weight, 'bias': bias}

    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1))

    return {
        'params': params,
        'feats': feats,
        'coords': coords,
        'shape': shape,
        'neighbor_cache': neighbor_cache,
    }


def igemm_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = allow_tf32

    # Create random weight and bias matrices.
    params = torch.nn.ParameterDict()
    for i in range(L):
        weight = torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype))
        bias = torch.nn.Parameter(torch.randn(C, device=feats.device, dtype=feats.dtype))
        params[f'layer{i}'] = {'weight': weight, 'bias': bias}

    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1))

    return {
        'params': params,
        'feats': feats,
        'coords': coords,
        'shape': shape,
        'neighbor_cache': neighbor_cache,
    }


def igemm_splitk_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM_SPLITK)
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = allow_tf32

    # Create random weight and bias matrices.
    params = torch.nn.ParameterDict()
    for i in range(L):
        weight = torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype))
        bias = torch.nn.Parameter(torch.randn(C, device=feats.device, dtype=feats.dtype))
        params[f'layer{i}'] = {'weight': weight, 'bias': bias}

    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1))

    return {
        'params': params,
        'feats': feats,
        'coords': coords,
        'shape': shape,
        'neighbor_cache': neighbor_cache,
    }


def migemm_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM)
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = allow_tf32

    # Create random weight and bias matrices.
    params = torch.nn.ParameterDict()
    for i in range(L):
        weight = torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype))
        bias = torch.nn.Parameter(torch.randn(C, device=feats.device, dtype=feats.dtype))
        params[f'layer{i}'] = {'weight': weight, 'bias': bias}

    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1))

    return {
        'params': params,
        'feats': feats,
        'coords': coords,
        'shape': shape,
        'neighbor_cache': neighbor_cache,
    }

def migemm_splitk_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = allow_tf32

    # Create random weight and bias matrices.
    params = torch.nn.ParameterDict()
    for i in range(L):
        weight = torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype))
        bias = torch.nn.Parameter(torch.randn(C, device=feats.device, dtype=feats.dtype))
        params[f'layer{i}'] = {'weight': weight, 'bias': bias}

    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1))

    return {
        'params': params,
        'feats': feats,
        'coords': coords,
        'shape': shape,
        'neighbor_cache': neighbor_cache,
    }


def flex_gemm_kernel_fn(params, feats, coords, shape, neighbor_cache):
    zero_grad(params.parameters())
    h = feats
    for i in range(len(params)):
        weight = params[f'layer{i}']['weight']
        bias = params[f'layer{i}']['bias']
        h = sparse_submanifold_conv3d(h, coords, shape, weight, bias, neighbor_cache)[0]
    h.backward(torch.ones_like(h))


def test_conv_fwd():
    # Matrix dimensions.
    config = [
        {'RES': 8, 'C': 1024, 'L': 16},
        {'RES': 16, 'C': 1024, 'L': 8},
        {'RES': 32, 'C': 1024, 'L': 4},
        {'RES': 64, 'C': 1024, 'L': 2},
        {'RES': 128, 'C': 512, 'L': 2},
        {'RES': 256, 'C': 256, 'L': 2},
        # {'RES': 512, 'C': 128, 'L': 2},
        # {'RES': 1024, 'C': 64, 'L': 2},
        # {'RES': 2048, 'C': 64, 'L': 1},
    ]
    
    # List of custom kernel functions.
    kernel_functions = {
        'egemm': (flex_gemm_kernel_fn, egemm_prepare_fn),
        'igemm': (flex_gemm_kernel_fn, igemm_prepare_fn),
        'igemm_splitk': (flex_gemm_kernel_fn, igemm_splitk_prepare_fn),
        'migemm': (flex_gemm_kernel_fn, migemm_prepare_fn),
        'migemm_splitk': (flex_gemm_kernel_fn, migemm_splitk_prepare_fn),
    }
        
    results = {}
    for c in tqdm(config, leave=False):
        RES, C, L = c['RES'], c['C'], c['L']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, dtype=DTYPE)
        args = {
            'feats': feats,
            'coords': coords,
            'shape': shape,
            'RES': RES,
            'C': C,
            'L': L,
        }

        config_key = f'RES={RES},C={C},L={L}'
        results[config_key] = {
            'time': [],
            'memory': [],
        }

        # Benchmark each custom kernel.
        for kernel_fn, prepare_fn in kernel_functions.values():
            avg_time, memory, C_kernel = benchmark_kernel(kernel_fn, **args, prepare_fn=prepare_fn)
            if memory != 'OOM':
                results[config_key]['time'].append(f'{avg_time:.2f} ms')
                results[config_key]['memory'].append(f'{memory:.1f}G')
            else:
                results[config_key]['time'].append('N/A')
                results[config_key]['memory'].append('OOM')
                
    # Print results as a formatted table.
    print("=" * 180)
    print("Conv Forward Benchmark Results")
    print("=" * 180)
    for m in ['time','memory']:
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

    configs = [f"{c['RES']},{c['C']},{c['L']}" for c in config]
    kernels = list(kernel_functions.keys())
    x = np.arange(len(configs))
    width = 0.16

    plt.figure(figsize=(12.5, 5))
    plt.subplot(2, 1, 1)
    plt.ylabel("Relative Speed")
    for i, k in enumerate(kernels):
        min_time = [float(v['time'][-1].split()[0]) for v in results.values()]
        times = [min_time[j] / float(v['time'][i].split()[0]) for j, v in enumerate(results.values())]
        bars = plt.bar(x + i * width, times, width - 0.02, label=k, edgecolor="black", linewidth=0.8, zorder=2)

        for bar, time in zip(bars, times):
            if i == len(kernels) - 1:
                bar.set_facecolor("#20a3c4")
            else:
                gray = 0.8 - 0.6 * (i / (5 - 2))
                bar.set_facecolor(str(gray))

            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{time:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    plt.xlabel("Configuration (RES, C, L)")
    plt.xticks(x + width * (len(kernels) - 1) / 2, configs, ha="center")
    plt.tick_params(axis="x", length=0)
    plt.yticks([])
    plt.tick_params(axis="y", length=0)
    plt.legend()
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )
    plt.ylim(0, 1.2)
    for y in np.arange(0, 1.2, 0.2):
        plt.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

    plt.subplot(2, 1, 2)
    plt.ylabel("Relative VRAM")
    for i, k in enumerate(kernels):
        min_memory = [float(v['memory'][-1].split('G')[0]) for v in results.values()]
        memorys = [float(v['memory'][i].split('G')[0]) / min_memory[j] for j, v in enumerate(results.values())]
        bars = plt.bar(x + i * width, memorys, width - 0.02, label=k, edgecolor="black", linewidth=0.8, zorder=2)

        for bar, memory in zip(bars, memorys):
            if i == len(kernels) - 1:
                bar.set_facecolor("#20a3c4")
            else:
                gray = 0.8 - 0.6 * (i / (5 - 2))
                bar.set_facecolor(str(gray))

            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{memory:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    plt.xlabel("Configuration (RES, C, L)")
    plt.xticks(x + width * (len(kernels) - 1) / 2, configs, ha="center")
    plt.tick_params(axis="x", length=0)
    plt.yticks([])
    plt.tick_params(axis="y", length=0)
    plt.legend()
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )
    ymax = plt.ylim()[1]
    plt.ylim(0, ymax+0.5)
    for y in np.arange(0, ymax + 0.5, 1):
        plt.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

    plt.tight_layout()
    plt.savefig(f"conv_fwd.png", dpi=300)


if __name__ == "__main__":
    test_conv_fwd()
