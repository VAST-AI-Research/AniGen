from tqdm import tqdm
import torch
import flex_gemm
from flex_gemm.ops.spconv import SubMConv3dFunction
from utils import sphere_coords, calc_err, benchmark_kernel


def spconv_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor, **kwargs):
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    
    # Init module.
    module = spconv.SubMConv3d(Ci, Co, ksize, indice_key='test', algo=spconv.ConvAlgo.MaskSplitImplicitGemm).cuda().to(feats.dtype)
    module.weight.data.copy_(weight)
    module.bias.data.copy_(bias)
    
    # Init input tensor and its cache
    input_spconv = spconv.SparseConvTensor(feats, coords, shape[-3:], shape[0])
    out_spconv = module(input_spconv)
    input_spconv.indice_dict = out_spconv.indice_dict.copy()
    
    return {
        'module': module,
        'input': input_spconv,
    }
    

def spconv_kernel_fn(module, input):
    return module(input).features


def torchsparse_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor, **kwargs):
    conv_config = torchsparse.nn.functional.conv_config.get_default_conv_config()
    torchsparse.nn.functional.conv_config.set_global_conv_config(conv_config)
    torchsparse.backends.benchmark = True

    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    
    # Init module.
    module = torchsparse.nn.Conv3d(Ci, Co, ksize, bias=True).cuda().to(feats.dtype)
    module.kernel.data.copy_(weight.permute(3, 2, 1, 4, 0).reshape(-1, Ci, Co).contiguous())  # torchsparse uses (V, Cin, Cout) format
    module.bias.data.copy_(bias)
    
    # Init input tensor
    input_torchsparse = torchsparse.SparseTensor(feats, coords, spatial_range=[shape[0],*shape[-3:]])
    out_torchsparse = module(input_torchsparse)
    input_torchsparse._caches = out_torchsparse._caches
    
    return {
        'module': module,
        'input': input_torchsparse,
    }


def torchsparse_kernel_fn(module, input):
    return module(input).feats


def fvdb_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor, **kwargs):
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])

    weight = weight.permute(0, 4, 3, 2, 1).contiguous()
    
    grid = fvdb.gridbatch_from_ijk(coords[:, 1:].contiguous(), voxel_sizes=0.01)
    input = grid.jagged_like(feats)
    sparse_conv_packinfo, out_grid = grid.sparse_conv_kernel_map(kernel_size=ksize, stride=1)
    sparse_conv_packinfo.build_implicit_gemm(
        sorted=True, split_mask_num=1, training=True, split_mask_num_bwd=3, use_tf32=True
    )

    return {
        'input': input,
        'weight': weight,
        'bias': bias,
        'sparse_conv_packinfo': sparse_conv_packinfo,
    }


def fvdb_kernel_fn(input, weight, bias, sparse_conv_packinfo):
    output = sparse_conv_packinfo.sparse_conv_3d(input, weights=weight, backend=fvdb.ConvPackBackend.IGEMM).jflatten().jdata + bias
    return output


def torch_theory_kernel_fn(A, B, bias):
    C = torch.addmm(bias, A, B.T)
    return None


def torch_theory_all_prepare_fn(feats: torch.Tensor, weight: torch.Tensor, **kwargs):
    N, Ci, Co = feats.shape[0], weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    V = ksize[0] * ksize[1] * ksize[2]
    A = torch.randn((N, V * Ci), device=feats.device, dtype=feats.dtype)
    B = torch.randn((Co, V * Ci), device=feats.device, dtype=feats.dtype)
    bias = torch.randn(Co, device=feats.device, dtype=feats.dtype)
    return {
        'A': A,
        'B': B,
        'bias': bias,
    }
    

def torch_theory_req_prepare_fn(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size, weight: torch.Tensor, **kwargs):
    Ci, Co = weight.shape[-1], weight.shape[0]
    ksize = (weight.shape[1], weight.shape[2], weight.shape[3])
    dilation = (1, 1, 1)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    L = (neighbor_cache['neighbor_map']!=0xffffffff).sum()
    A = torch.randn((L, Ci), device=feats.device, dtype=feats.dtype)
    B = torch.randn((Co, Ci), device=feats.device, dtype=feats.dtype)
    bias = torch.randn(Co, device=feats.device, dtype=feats.dtype)
    return {
        'A': A,
        'B': B,
        'bias': bias,
    }


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


def test_conv_fwd():
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
        # 'torch_all_ref': (torch_theory_kernel_fn, torch_theory_all_prepare_fn),
        # 'torch_req_ref': (torch_theory_kernel_fn, torch_theory_req_prepare_fn),
        # 'spconv': (spconv_kernel_fn, spconv_prepare_fn),
        # 'torchsparse': (torchsparse_kernel_fn, torchsparse_prepare_fn),
        # 'fvdb': (fvdb_kernel_fn, fvdb_prepare_fn),
        'egemm': (SubMConv3dFunction._sparse_submanifold_conv_forward, egemm_prepare_fn),
        'igemm': (SubMConv3dFunction._sparse_submanifold_conv_forward, igemm_prepare_fn),
        'igemmk': (SubMConv3dFunction._sparse_submanifold_conv_forward, igemmk_prepare_fn),
        'migemm': (SubMConv3dFunction._sparse_submanifold_conv_forward, migemm_prepare_fn),
        'migemmk': (SubMConv3dFunction._sparse_submanifold_conv_forward, migemmk_prepare_fn),
    }
    
    reference = (SubMConv3dFunction._sparse_submanifold_conv_forward, egemm_prepare_fn)
    
    results = {}
    for c in tqdm(config, leave=False):
        RES, C = c['RES'], c['C']

        # Create random input matrices.
        feats, coords, shape = sphere_coords(RES, C, dtype=torch.float16)
        weight = torch.randn(C, 3, 3, 3, C, device=feats.device, dtype=feats.dtype)
        bias = torch.randn(C, device=feats.device, dtype=feats.dtype)
        args = {
            'feats': feats,
            'coords': coords,
            'shape': shape,
            'weight': weight,
            'bias': bias,
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
    print("Conv Forward Benchmark Results")
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
    test_conv_fwd()
