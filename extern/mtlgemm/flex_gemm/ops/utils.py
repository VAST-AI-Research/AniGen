import torch


def init_hashmap(spatial_size, hashmap_size, device):
    N, C, W, H, D = spatial_size
    VOL = N * W * H * D

    # If the number of elements in the tensor is less than 2^32, use uint32 as the hashmap type, otherwise use uint64.
    if VOL < 2**32:
        keys_dtype = torch.uint32
        keys_fill = torch.iinfo(torch.uint32).max
    elif VOL < 2**64:
        keys_dtype = torch.uint64
        keys_fill = torch.iinfo(torch.uint64).max
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Get volumn {VOL} > 2^64.")

    # MPS doesn't have a kernel for torch.full / torch.empty on uint32/uint64
    # tensors yet (DispatchStub: missing kernel for mps). Stage the allocation
    # through CPU and transfer — `.to('mps')` for these dtypes does work, and
    # the underlying MTLBuffer storage is byte-identical.
    target_device = torch.device(device) if not isinstance(device, torch.device) else device
    if target_device.type == "mps":
        hashmap_keys = torch.full((hashmap_size,), keys_fill, dtype=keys_dtype).to(target_device)
        hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32).to(target_device)
    else:
        hashmap_keys = torch.full((hashmap_size,), keys_fill, dtype=keys_dtype, device=target_device)
        hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32, device=target_device)

    return hashmap_keys, hashmap_vals
