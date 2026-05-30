from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import spconv as spconv_core
import spconv.pytorch as spconv
import torchsparse
import torchsparse.nn
import torchsparse.nn.functional
import fvdb
import warpconvnet.nn.modules.sparse_conv as warpconvnet_sparse_conv
import warpconvnet.geometry.types.voxels as warpconvnet_voxels
import flex_gemm
from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d
from utils import sphere_coords, benchmark_kernel, zero_grad


DTYPE = torch.float16
allow_tf32 = True


def spconv_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    spconv_core.constants.SPCONV_ALLOW_TF32 = allow_tf32
    # Init module.
    model = torch.nn.ModuleList([
        spconv.SubMConv3d(C, C, 3, algo=spconv.ConvAlgo.MaskSplitImplicitGemm).cuda().to(feats.dtype)
        for _ in range(L)
    ])
   
    return {
        'model': model,
        'feats': feats,
        'coords': coords,
        'shape': shape,
    }


def spconv_kernel_fn(model, feats, coords, shape):
    zero_grad(model.parameters())
    h = spconv.SparseConvTensor(feats, coords, shape[-3:], shape[0])
    for layer in model:
        h = layer(h)
    h = h.features
    h.backward(torch.ones_like(h))


def torchsparse_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    torchsparse.backends.allow_tf32 = allow_tf32
    torchsparse.backends.hash_rsv_ratio = 4
    torchsparse.nn.functional.set_kmap_mode("hashmap_on_the_fly")
    conv_mode = torchsparse.nn.functional.ConvMode.mode1
    ts_cfg = torchsparse.nn.functional.conv_config.get_default_conv_config(conv_mode=conv_mode, training=False)
    ts_cfg.dataflow = torchsparse.nn.functional.Dataflow.ImplicitGEMM

    # Init module.
    model = torch.nn.ModuleList([
        torchsparse.nn.Conv3d(C, C, 3, bias=True).cuda().to(feats.dtype)
        for _ in range(L)
    ])
    
    return {
        'model': model,
        'feats': feats,
        'coords': coords,
        'shape': shape,
    }


def torchsparse_kernel_fn(model, feats, coords, shape):
    zero_grad(model.parameters())
    h = torchsparse.SparseTensor(feats, coords, spatial_range=[shape[0],*shape[-3:]])
    for layer in model:
        h = layer(h)
    h = h.feats
    h.backward(torch.ones_like(h))


def fvdb_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    # Init module.
    model = torch.nn.ModuleList([
        fvdb.nn.SparseConv3d(C, C, 3, bias=True).cuda().to(feats.dtype)
        for _ in range(L)
    ])

    for layer in model:
        layer.allow_tf32 = allow_tf32
    
    return {
        'model': model,
        'feats': feats,
        'coords': coords[:, 1:].contiguous().long(),
        'shape': shape,
    }


def fvdb_kernel_fn(model, feats, coords, shape):
    zero_grad(model.parameters())
    grid = fvdb.gridbatch_from_ijk(coords)
    h = fvdb.nn.VDBTensor(grid, grid.jagged_like(feats))
    for layer in model:
        h = layer(h)
    h = h.data.jdata
    h.backward(torch.ones_like(h))


def warpconvnet_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    # Init module.
    model = torch.nn.ModuleList([
        warpconvnet_sparse_conv.SparseConv3d(C, C, 3, bias=True).cuda().to(feats.dtype)
        for _ in range(L)
    ])
    
    return {
        'model': model,
        'feats': feats,
        'coords': coords[:, 1:].contiguous().int(),
        'shape': shape,
    }


def warpconvnet_kernel_fn(model, feats, coords, shape):
    zero_grad(model.parameters())
    h = warpconvnet_voxels.Voxels([coords], [feats])
    for layer in model:
        h = layer(h)
    h = h.feature_tensor
    h.backward(torch.ones_like(h))


def flex_gemm_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, RES, C, L):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = allow_tf32

    # Create random weight and bias matrices.
    params = torch.nn.ParameterDict()
    for i in range(L):
        weight = torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype))
        bias = torch.nn.Parameter(torch.randn(C, device=feats.device, dtype=feats.dtype))
        params[f'layer{i}'] = {'weight': weight, 'bias': bias}

    return {
        'params': params,
        'feats': feats,
        'coords': coords,
        'shape': shape,
    }


def flex_gemm_kernel_fn(params, feats, coords, shape):
    zero_grad(params.parameters())
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1))
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
        {'RES': 512, 'C': 128, 'L': 2},
        {'RES': 1024, 'C': 64, 'L': 2},
    ]
    
    # List of custom kernel functions.
    kernel_functions = {
        'fvdb': (fvdb_kernel_fn, fvdb_prepare_fn),
        'torchsparse': (torchsparse_kernel_fn, torchsparse_prepare_fn),
        'warpconvnet': (warpconvnet_kernel_fn, warpconvnet_prepare_fn),
        'spconv': (spconv_kernel_fn, spconv_prepare_fn),
        'flex_gemm': (flex_gemm_kernel_fn, flex_gemm_prepare_fn),
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
            results[config_key]['time'].append(f'{avg_time:.2f} ms')
            results[config_key]['memory'].append(f'{memory:.1f}G')
                
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

    plt.figure(figsize=(12.5, 2.5))

    for i, k in enumerate(kernels):
        min_time = [min([float(v['time'][j].split()[0]) for j in range(len(v['time']))]) for v in results.values()]
        times = [min_time[j] / float(v['time'][i].split()[0]) for j, v in enumerate(results.values())]
        bars = plt.bar(x + i * width, times, width - 0.02, label=k, edgecolor="black", linewidth=0.8, zorder=2)

        for bar, time in zip(bars, times):
            if i == len(kernels) - 1:
                bar.set_facecolor("#20a3c4")
            else:
                gray = 0.8 - 0.6 * (i / (len(kernels) - 2))
                bar.set_facecolor(str(gray))

            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{time:.2f}',
                ha='center',
                va='bottom',
                fontsize=7
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
    plt.tight_layout()
    plt.savefig(f"conv_fwd.png", dpi=300)


if __name__ == "__main__":
    test_conv_fwd()
