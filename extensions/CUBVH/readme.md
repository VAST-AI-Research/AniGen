# cuBVH

A CUDA Mesh BVH acceleration toolkit.

### Highlights

- Build acceleration structures from `numpy` or `torch` arrays on either CPU or CUDA.
- Save and reload BVHs without rebuilding via `torch.save`, `torch.load`, or `export_state`.
- Move a BVH between any Torch-visible devices with `.to(device)`, allocating GPU memory lazily.
- Run multi-GPU workloads safely; internal device guards keep each query on the intended GPU.

### Install

Make sure `torch` and CUDA are installed first.

```bash
pip install git+https://github.com/yihua7/cubvh --no-build-isolation

# or locally
git clone --recursive https://github.com/yihua7/cubvh
cd cubvh
pip install . --no-build-isolation
```
It will take several minutes to build the CUDA dependency.

#### Trouble Shooting
**`fatal error: eigen/matrix.h: No such file or directory`**

This is a known issue for `torch==2.1.0` and `torch==2.1.1` (https://github.com/pytorch/pytorch/issues/112841). 
To patch up these two versions, clone this repository, and copy `patch/eigen` to your pytorch include directory:
```bash
# for example, if you are using anaconda (assume base env)
cp -r patch/eigen ~/anaconda3/lib/python3.9/site-packages/torch/include/pybind11/
```

**`fatal error: Eigen/Dense: No such file or directory`**

Please make sure [`eigen >= 3.3`](https://eigen.tuxfamily.org/index.php?title=Main_Page) is installed. 
We have included it as a submodule in this repository, but you can also install it in your system include path.
(For example, ubuntu systems can use `sudo apt install libeigen3-dev`.)

### Usage

**Basics:**

```python
import numpy as np
import trimesh
import torch

import cubvh

### build BVH from mesh
mesh = trimesh.load('example.ply')
# NOTE: you need to normalize the mesh first, since the max distance is hard-coded to 100.
BVH = cubvh.cuBVH(mesh.vertices, mesh.faces) # build with numpy.ndarray/torch.Tensor

### query ray-mesh intersection
rays_o, rays_d = get_ray(pose, intrinsics, H, W) # [N, 3], [N, 3], query with torch.Tensor (cuda)
intersections, face_id, depth = BVH.ray_trace(rays_o, rays_d) # [N, 3], [N,], [N,]

### query unsigned distance
points # [N, 3]
# uvw is the barycentric corrdinates of the closest point on the closest face (None if `return_uvw` is False).
distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True) # [N], [N], [N, 3]

### query signed distance (INNER is NEGATIVE!)
# for watertight meshes (default)
distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='watertight') # [N], [N], [N, 3]
# for non-watertight meshes:
distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='raystab') # [N], [N], [N, 3]
```

**Serialization and Device Placement:**

cuBVH behaves like a Torch module: you can checkpoint it, reload it later, and migrate the
instance between CPU and CUDA devices without rebuilding the acceleration structure.

```python
import torch
import trimesh
import cubvh

mesh = trimesh.load('example.ply')
vertices, triangles = mesh.vertices, mesh.faces

# build directly on a target device (defaults to CUDA when available)
bvh = cubvh.cuBVH(vertices=vertices, triangles=triangles, device='cuda')

# move between CPU and CUDA like any other torch module
bvh_cpu = bvh.to('cpu')
bvh_cuda = bvh_cpu.to('cuda:0')

# persist to disk via torch.save / torch.load
torch.save(bvh_cuda, "mesh_bvh.pt")
reloaded = torch.load("mesh_bvh.pt")  # restored on CPU by default
reloaded = reloaded.to('cuda')         # lazily reinstantiates GPU buffers

# direct access to CPU copies of the acceleration data
triangles_np = reloaded.triangles_cpu           # (N, 3, 3) float32 array
nodes_np = reloaded.bvh_nodes_cpu               # dict with mins/maxs/children arrays
state_dict = reloaded.export_state()            # tensors for custom checkpoints
```

When a serialized BVH is first loaded, it resides on the CPU. Calling `.to('cuda:N')` lazily
reinstantiates the GPU buffers on the requested device and keeps subsequent queries there.
Advanced users can stash the `export_state()` tensors for custom checkpoint formats or stream
them to other processes.

**Robust Mesh Occupancy:**

UDF + flood-fill for possibly non-watertight/single-layer meshes:

```python
import torch
import cubvh
import numpy as np

resolution = 512
device = torch.device('cuda')

BVH = cubvh.cuBVH(vertices, faces)

grid_points = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, resolution, device=device),
        torch.linspace(-1, 1, resolution, device=device),
        torch.linspace(-1, 1, resolution, device=device),
        indexing="ij",
    ), dim=-1,
) # [N, N, N, 3]

# query dense UDF
udf, _, _ = BVH.unsigned_distance(grid_points.view(-1, 3), return_uvw=False)
udf = udf.view(opt.res, opt.res, opt.res).contiguous()

# floodfill to get SDF
occ = udf < 2 / resolution # tolerance 2 voxel
floodfill_mask = cubvh.floodfill(occ)
empty_label = floodfill_mask[0, 0, 0].item()
empty_mask = (floodfill_mask == empty_label)
occ_mask = ~empty_mask
sdf = udf - eps  # inner is negative
inner_mask = occ_mask & (sdf > 0)
sdf[inner_mask] *= -1

sdf = sdf.cpu().numpy()

```
Check [`test/extract_mesh_watertight.py`](test/extract_mesh_watertight.py) for more details.


### Acknowledgement

* This project is forked from [ashawkey/cubvh](https://github.com/ashawkey/cubvh) by [Jiaxiang Tang](https://github.com/ashawkey).
* Credits to [Thomas Müller](https://tom94.net/)'s amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp)!

### License

This project contains code under multiple licenses:
- Overall cubvh code: MIT License (see [LICENSE](LICENSE))
- BVH code derived from instant-ngp: [NVIDIA Source Code License (Non-Commercial)](LICENSE_NVIDIA)
- `gpu_memory.h`: BSD-3-Clause (NVIDIA/tiny-cuda-nn)
- `pcg32.h`: Apache 2.0

**Note:** The NVIDIA license restricts use to non-commercial / research purposes only.
