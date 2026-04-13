# cuBVH State & Device Management Update Log

This document describes the code changes that enable cuBVH instances to be serialized with `torch.save`, restored with `torch.load`, moved across devices via `.to(...)`, and released cleanly when they go out of scope.

## Python Front-End (`cubvh/api.py`)

- Replaced the eager GPU-only implementation handle with a lazily-instantiated backend plus a CPU-resident state cache (`self._state`).
- Added `_pull_state_from_impl()` to copy triangles, triangle ids, and BVH node tensors from CUDA memory to CPU tensors immediately after BVH construction.
- Added `_load_state()` to reconstruct the CPU state from serialized tensors, validating required keys and type expectations.
- Introduced `_instantiate_impl()` and `_release_impl()` to create and destroy the CUDA backend on demand, ensuring GPU memory is only held when the BVH resides on a CUDA device.
- Implemented `to(...)` to support `cpu` ⇄ `cuda` transfers. Moving to CUDA reconstructs the backend from the cached CPU tensors; moving to CPU frees the CUDA handle.
- Added safeguards so every query (`ray_trace`, `unsigned_distance`, `signed_distance`) ensures the BVH lives on a CUDA device before use.
- Exposed convenience accessors (`triangles_cpu`, `bvh_nodes_cpu`) and `export_state()` for retrieving CPU copies of the serialized data.
- Implemented Python-side pickling helpers (`__getstate__`, `__setstate__`) leveraging the new CPU state, enabling `torch.save` / `torch.load` round-trips.
- Wrapped all CUDA interactions within a `_cuda_device_guard` helper so building, serialization, and query kernels execute on the cuBVH instance's own `device`, eliminating the need for callers to manage global CUDA context and fixing multi-threaded multi-GPU usage.

## CUDA Binding Layer (`src/api_gpu.cu`, `include/gpu/api_gpu.h`, `src/bindings.cpp`)

- Augmented `cuBVH::export_state()` to emit tensors for triangle vertices/ids and the BVH node bounding boxes & child indices.
- Added a constructor overload that recreates a CUDA BVH directly from CPU vectors of `Triangle` and `TriangleBvhNode` objects, bypassing the costly rebuild step.
- Plumbed new C++ factory bindings: `create_cuBVH_from_state(...)` mirrors the CPU tensors produced by `export_state()` so Python can lazily rebuild GPU state on demand.
- Updated `include/gpu/api_gpu.h` with the new factory signature and exposed host-node accessors via `TriangleBvh::host_nodes()` / `set_nodes(...)`.
- Ensured BVH node escape-link threading remains available by including `<functional>` in `src/bvh.cu`.

## BVH Core (`src/bvh.cu`, `include/gpu/bvh.cuh`)

- Exposed host node storage (`host_nodes()`) and setter (`set_nodes(...)`) to allow rehydrating the BVH structure from serialized data.
- Guaranteed that GPU buffers mirror the host nodes after both rebuilds and deserialization by invoking `resize_and_copy_from_host(...)` in the appropriate locations.
- Added the missing `<functional>` include required for the threading lambda used during BVH construction.

## Testing (`test/test_bvh_serialization.py`)

- Added an executable test that constructs a BVH on CUDA, serializes it with `torch.save`, restores it via `torch.load`, and verifies ray distance results match bit-for-bit.
- Exercised device transitions (`cuda → cpu → cuda`) to confirm lazy instantiation works and GPU memory is released when the BVH is moved off-device.
- Validated exported numpy views (`triangles_cpu`, `bvh_nodes_cpu`) and state dictionary contents to ensure downstream tooling can consume serialized data.
- Added a multi-threaded multi-device smoke test (`test/test_bvh_thread_device.py`) that spawns one thread per GPU, builds cuBVH instances, moves them across devices, synchronizes results, and verifies no illegal memory access occurs, confirming the new device guard strategy works without manual CUDA context management.

## Documentation (`readme.md`)

- Documented the new `.to(device)` semantics, serialization workflow, and CPU export utilities with a concrete example using `trimesh`.

## Installation & Evaluation

- Rebuilt and installed the package locally using `pip install . --no-build-isolation` to pick up the updated extension code.
- Ran the new serialization test script to verify correctness on CUDA hardware.
