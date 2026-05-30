import time
from tqdm import tqdm
import torch
from flex_gemm import kernels
from utils import sphere_coords


@torch.no_grad()
def test_hashmap():
    RES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    records = {}
    
    test_cases = []
    for res in RES:
        test_cases.append((res, (torch.uint32, torch.uint32)))
        test_cases.append((res, (torch.uint32, torch.uint64)))
        test_cases.append((res, (torch.uint64, torch.uint32)))
        test_cases.append((res, (torch.uint64, torch.uint64)))
    
    for res, (dtype_key, dtype_value) in tqdm(test_cases, leave=False):
        time_insert = 0
        memory_insert = 0
        oom_insert = False
        cnt_insert = 0
        time_lookup = 0
        memory_lookup = 0
        oom_lookup = False
        cnt_lookup = 0
        feats, coords, shape = sphere_coords(res, 0)
        for i in range(110):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_insert = time.time()
            try:
                hashmap_keys = torch.full((2 * coords.shape[0],), torch.iinfo(dtype_key).max, dtype=dtype_key, device=coords.device)
                hashmap_values = torch.empty((2 * coords.shape[0],), dtype=dtype_value, device=coords.device)
                values = torch.randint(0, torch.iinfo(dtype_value).max//2, (coords.shape[0],), device=coords.device).to(dtype_value)
                kernels.cuda.hashmap_insert_3d_cuda(hashmap_keys, hashmap_values, coords, values, res, res, res)
            
                torch.cuda.synchronize()
                end_insert = time.time()
                if i > 10:
                    time_insert += end_insert - start_insert
                    memory_insert += torch.cuda.max_memory_allocated() / 1024**3
                    cnt_insert += 1
            except Exception as e:
                if 'out of memory' in str(e):
                    oom_insert = True
                else:
                    raise e
                
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_lookup = time.time()
            try:
                values_ = kernels.cuda.hashmap_lookup_3d_cuda(hashmap_keys, hashmap_values, coords, res, res, res)
                torch.cuda.synchronize()
                end_lookup = time.time()
                if i > 10:
                    time_lookup += end_lookup - start_lookup
                    memory_lookup += torch.cuda.max_memory_allocated() / 1024**3
                    cnt_lookup += 1
            except Exception as e:
                if 'out of memory' in str(e):
                    oom_lookup = True
                else:
                    raise e
                
            assert torch.all(values == values_), f"Hashmap lookup failed for res={res}, dtype_key={dtype_key}, dtype_value={dtype_value}"
            
        time_insert = f"{time_insert / cnt_insert:.5f}s" if cnt_insert > 0 else "N/A"
        memory_insert = "OOM" if oom_insert else f"{memory_insert / cnt_insert:.5f}G" if cnt_insert > 0 else "N/A"
        time_lookup = f"{time_lookup / cnt_lookup:.5f}s" if cnt_lookup > 0 else "N/A"
        memory_lookup = "OOM" if oom_lookup else f"{memory_lookup / cnt_lookup:.5f}G" if cnt_lookup > 0 else "N/A"
        records[f"res={res}, dtype_key={dtype_key}, dtype_value={dtype_value}"] = {
            'insert': f"{time_insert} | {memory_insert}",
            'lookup': f"{time_lookup} | {memory_lookup}",
        }
        res_str = []
        res_str.append("=" * 120)
        res_str.append(f"Hashmap test")
        res_str.append("=" * 120)
        res_str.append(f"{'settings':<64}{'insert':<24}{'lookup':<24}")
        for key, value in records.items():
            res_str.append(f"{key:<64}{value['insert']:<24}{value['lookup']:<24}")
    
    print('\n'.join(res_str))
    return res_str


all_tests = {
    'test_hashmap': test_hashmap,
}
                    

if __name__ == '__main__':
    for test_name, test_func in all_tests.items():
        records = test_func()
