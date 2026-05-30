import time
from tqdm import tqdm
import torch
from flex_gemm import ops
from utils import sphere_coords


@torch.no_grad()
def test_hashmap():
    RES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    records = {}
    
    test_cases = []
    for res in RES:
        test_cases.append((res, 'z_order', 'cpu'))
        test_cases.append((res, 'z_order', 'cuda'))
        test_cases.append((res, 'hilbert', 'cpu'))
        test_cases.append((res, 'hilbert', 'cuda'))
    
    for res, mode, device in tqdm(test_cases, leave=False):
        time_enc = 0
        memory_enc = 0
        oom_enc = False
        cnt_enc = 0
        time_dec = 0
        memory_dec = 0
        oom_dec = False
        cnt_dec = 0
        feats, coords, shape = sphere_coords(res, 0)
        coords = coords.to(device)
        for i in range(110):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_enc = time.time()
            try:
                codes = ops.serialize.encode_seq(coords, shape, mode=mode)
                torch.cuda.synchronize()
                end_enc = time.time()
                if i > 10:
                    time_enc += end_enc - start_enc
                    memory_enc += torch.cuda.max_memory_allocated() / 1024**3
                    cnt_enc += 1
            except Exception as e:
                if 'out of memory' in str(e):
                    oom_enc = True
                else:
                    raise e
                
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_dec = time.time()
            try:
                coords_ = ops.serialize.decode_seq(codes, shape, mode=mode)
                torch.cuda.synchronize()
                end_dec = time.time()
                if i > 10:
                    time_dec += end_dec - start_dec
                    memory_dec += torch.cuda.max_memory_allocated() / 1024**3
                    cnt_dec += 1
            except Exception as e:
                if 'out of memory' in str(e):
                    oom_dec = True
                else:
                    raise e
                
            assert torch.all(coords == coords_), f"Serialization failed for res={res}, mode={mode}"
            
        time_enc = f"{time_enc / cnt_enc:.5f}s" if cnt_enc > 0 else "N/A"
        memory_enc = "OOM" if oom_enc else f"{memory_enc / cnt_enc:.5f}G" if cnt_enc > 0 else "N/A"
        time_dec = f"{time_dec / cnt_dec:.5f}s" if cnt_dec > 0 else "N/A"
        memory_dec = "OOM" if oom_dec else f"{memory_dec / cnt_dec:.5f}G" if cnt_dec > 0 else "N/A"
        records[f"res={res}, mode={mode}, device={device}"] = {
            'encode': f"{time_enc} | {memory_enc}",
            'decode': f"{time_dec} | {memory_dec}",
        }
        res_str = []
        res_str.append("=" * 120)
        res_str.append(f"Serialize test")
        res_str.append("=" * 120)
        res_str.append(f"{'settings':<64}{'encode':<24}{'decode':<24}")
        for key, value in records.items():
            res_str.append(f"{key:<64}{value['encode']:<24}{value['decode']:<24}")
    
    print('\n'.join(res_str))
    return res_str


all_tests = {
    'test_hashmap': test_hashmap,
}
                    

if __name__ == '__main__':
    for test_name, test_func in all_tests.items():
        records = test_func()
