"""Apple Silicon bootstrap for AniGen inference.

Import and call configure_mps_environment() BEFORE importing anigen.* so that
the sparse-conv / attention backend env vars are read at module import time.
"""
import os


def configure_mps_environment() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Dense attention: AniGen's 'naive' path is real fp32 matmul+softmax (not SDPA). Safe on MPS.
    os.environ.setdefault("ATTN_BACKEND", "naive")
    # Sparse attention: route to our fp32 fallback (Task 7). MPS fused SDPA is banned
    # (the >~18-20k-token cliff returns catastrophically wrong output — proven in Pixal3D).
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
    # Sparse conv: select the spconv-family module (basic.py), but route the actual
    # kernel to flex_gemm via SPARSE_CONV_BACKEND (Task 9 reads this).
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("SPCONV_ALGO", "native")
    # Naive (full) sparse attention over the SLAT token set is O(N^2). At SLAT
    # resolution N is tens of thousands of tokens, so a single score tensor
    # [H, q_chunk, N] is multi-GB and copies/softmaxes pathologically slowly on
    # MPS (observed: a single _to_copy stalling SLAT step 0 indefinitely). Bound
    # peak memory by tiling the KEY dimension with an online softmax (math is
    # identical). 4096 keeps the score tile small while limiting Python-loop
    # overhead. CUDA never imports anigen_mps, so its attention is unchanged.
    os.environ.setdefault("ANIGEN_ATTN_KEY_CHUNK", "4096")


def resolve_device(requested: str = "mps"):
    import torch
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[anigen_mps] MPS unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def install_nvdiffrast_alias() -> None:
    """Make `import nvdiffrast.torch` resolve to mtldiffrast (Metal) on non-CUDA.

    utils3d.torch.rasterization hard-imports nvdiffrast.torch at module top;
    mtldiffrast is the API-compatible Metal port. We expose the nvdiffrast
    context-class names utils3d's RastContext instantiates
    (RasterizeGLContext for the default backend='gl', RasterizeCudaContext for
    backend='cuda') as aliases of MtlRasterizeContext so RastContext
    construction works unchanged, and re-export rasterize/interpolate/antialias/
    texture from mtldiffrast.
    """
    import sys, types
    try:
        import torch
        if torch.cuda.is_available():
            return  # real nvdiffrast path on CUDA boxes; do not shim
    except Exception:
        pass
    try:
        import mtldiffrast.torch as _mdr
    except Exception:
        return  # mtldiffrast unavailable; leave nvdiffrast absent (errors loudly if used)
    if "nvdiffrast.torch" in sys.modules:
        return
    nvd = types.ModuleType("nvdiffrast")
    nvd_torch = types.ModuleType("nvdiffrast.torch")
    # re-export everything mtldiffrast.torch provides (rasterize, interpolate,
    # antialias, texture, DepthPeeler, MtlRasterizeContext, ...).
    for _n in dir(_mdr):
        if not _n.startswith("__"):
            setattr(nvd_torch, _n, getattr(_mdr, _n))
    # nvdiffrast context-class names utils3d's RastContext expects -> Metal context.
    for _alias in ("RasterizeCudaContext", "RasterizeGLContext"):
        if not hasattr(nvd_torch, _alias):
            setattr(nvd_torch, _alias, _mdr.MtlRasterizeContext)
    # valid specs so importlib.util.find_spec won't choke (e.g. conftest stub finder).
    import importlib.machinery as _m
    nvd.__spec__ = _m.ModuleSpec("nvdiffrast", loader=None, is_package=True)
    nvd.__path__ = []
    nvd_torch.__spec__ = _m.ModuleSpec("nvdiffrast.torch", loader=None)
    nvd.torch = nvd_torch
    sys.modules["nvdiffrast"] = nvd
    sys.modules["nvdiffrast.torch"] = nvd_torch


def install_knn_shim() -> None:
    """Replace pytorch3d.ops.knn_points/ball_query with CPU cKDTree drop-ins.

    pytorch3d's MPS backend for these ops is broken/absent. Install (or patch) a
    pytorch3d.ops module BEFORE anigen imports it so call sites bind to ours.
    """
    import sys, types
    from importlib.machinery import ModuleSpec
    from anigen_mps import knn_cpu
    try:
        import pytorch3d.ops as _ops      # real package present -> patch in place
        _ops.knn_points = knn_cpu.knn_points
        _ops.ball_query = knn_cpu.ball_query
    except Exception:
        pkg = sys.modules.get("pytorch3d") or types.ModuleType("pytorch3d")
        if not hasattr(pkg, "__path__"):
            pkg.__path__ = []  # mark as a package so submodule imports proceed
        # Give the synthetic modules a real __spec__ so importlib.util.find_spec()
        # (e.g. the conftest stub-finder) doesn't choke on a None spec.
        if getattr(pkg, "__spec__", None) is None:
            pkg.__spec__ = ModuleSpec("pytorch3d", loader=None, is_package=True)
        ops = types.ModuleType("pytorch3d.ops")
        ops.__spec__ = ModuleSpec("pytorch3d.ops", loader=None)
        ops.knn_points = knn_cpu.knn_points
        ops.ball_query = knn_cpu.ball_query
        pkg.ops = ops
        sys.modules["pytorch3d"] = pkg
        sys.modules["pytorch3d.ops"] = ops


def _remap_cuda_device(arg, target):
    """Map a 'cuda'/torch.device('cuda') argument onto `target`; pass others through."""
    import torch
    if isinstance(arg, str) and arg.split(":")[0] == "cuda":
        return target
    if isinstance(arg, torch.device) and arg.type == "cuda":
        return torch.device(target)
    return arg


class _CudaToActiveDevice:
    """Context manager: remap `.to('cuda')` calls to the active MPS/CPU device.

    The DSINE torch.hub `hubconf.py` hardcodes `model.to(torch.device("cuda"))` and
    `self.device = torch.device('cuda')`. On a non-CUDA box that raises
    "Torch not compiled with CUDA enabled". Rather than rebind `torch.device` (which
    breaks `isinstance(x, torch.device)` inside torch.load), we wrap nn.Module.to and
    Tensor.to to remap any cuda destination to the active device for the duration of the
    hub load. CUDA boxes never enter this path (guarded by torch.cuda.is_available()).
    """

    def __init__(self, device):
        self._target = device
        self._orig_module_to = None
        self._orig_tensor_to = None

    def __enter__(self):
        import torch
        target = self._target
        self._orig_module_to = torch.nn.Module.to
        self._orig_tensor_to = torch.Tensor.to
        orig_module_to = self._orig_module_to
        orig_tensor_to = self._orig_tensor_to

        def module_to(self, *args, **kwargs):
            if args:
                args = (_remap_cuda_device(args[0], target),) + tuple(args[1:])
            if "device" in kwargs:
                kwargs["device"] = _remap_cuda_device(kwargs["device"], target)
            return orig_module_to(self, *args, **kwargs)

        def tensor_to(self, *args, **kwargs):
            if args:
                args = (_remap_cuda_device(args[0], target),) + tuple(args[1:])
            if "device" in kwargs:
                kwargs["device"] = _remap_cuda_device(kwargs["device"], target)
            return orig_tensor_to(self, *args, **kwargs)

        torch.nn.Module.to = module_to
        torch.Tensor.to = tensor_to
        return self

    def __exit__(self, *exc):
        import torch
        if self._orig_module_to is not None:
            torch.nn.Module.to = self._orig_module_to
        if self._orig_tensor_to is not None:
            torch.Tensor.to = self._orig_tensor_to
        return False


def cuda_to_active_device(device):
    """Return a context manager mapping torch.device('cuda') -> `device` (Mac-only use)."""
    return _CudaToActiveDevice(device)


def upcast_pipeline_fp32(pipeline) -> None:
    """Upcast fp16 flow/VAE models to fp32 in-place (Mac-only).

    AniGen constructs the SS/SLAT flow models and the SS/SLAT decoders with
    use_fp16=True (per their config.json), which runs convert_to_fp16() in __init__
    and loads fp16 weights. MPS mishandles mixed fp16/fp32 matmuls. Each model exposes
    convert_to_fp32(); call it where available and otherwise .float() the module so the
    whole inference graph runs in fp32. No-op / never called on CUDA.
    """
    import torch
    if torch.cuda.is_available():
        return
    for name, model in getattr(pipeline, "models", {}).items():
        if not isinstance(model, torch.nn.Module):
            continue
        if hasattr(model, "convert_to_fp32"):
            try:
                model.convert_to_fp32()
            except Exception:
                pass
        # convert_to_fp32() on these models only touches the "torso"/known submodules
        # and can leave some skin/decoder submodules in fp16. A full .float() recursively
        # casts every remaining fp16 param/buffer to fp32 (idempotent) so the entire
        # inference graph is single-dtype — MPS will not tolerate mixed fp16/fp32 matmuls.
        model.float()
        # Reset use_fp16/dtype on EVERY submodule, not just the top-level model.
        # Nested submodules (e.g. the decoder's skin SkinModel) keep their own
        # self.dtype=fp16 and do `h.type(self.dtype)` on activations; with weights
        # now fp32 that yields an fp16 x fp32 matmul, which MPS aborts on
        # ("Destination/Accumulator different datatype"). Make those casts no-ops.
        for sub in model.modules():
            if getattr(sub, "use_fp16", None) is not None:
                try:
                    sub.use_fp16 = False
                except Exception:
                    pass
            if isinstance(getattr(sub, "dtype", None), torch.dtype):
                try:
                    sub.dtype = torch.float32
                except Exception:
                    pass


def install_cuda_redirect(target: str = "mps") -> None:
    """Permanent CUDA -> active-device compatibility shim (Mac-only).

    AniGen's inference/postprocess code hardcodes CUDA three ways a non-CUDA box can't
    satisfy: ``tensor.cuda()`` / ``module.cuda()`` methods, ``x.to('cuda')``, and factory
    calls like ``torch.tensor(..., device='cuda')`` (21+ sites). Editing every call site
    is brittle, so redirect all three to the active MPS device, and downcast float64
    (unsupported on MPS) to float32 in transit. Guarded: never patched when real CUDA
    is present.
    """
    import torch
    if torch.cuda.is_available():
        return
    dev = resolve_device(target)
    _orig_tensor_to = torch.Tensor.to
    _orig_module_to = torch.nn.Module.to

    def _is_cuda(d):
        return (isinstance(d, str) and d.split(":")[0] == "cuda") or \
               (isinstance(d, torch.device) and d.type == "cuda")

    # (1) .cuda() methods
    def _tensor_cuda(self, *a, **k):
        t = self.float() if self.dtype == torch.float64 else self
        return _orig_tensor_to(t, dev)

    def _module_cuda(self, *a, **k):
        return _orig_module_to(self, dev)

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda

    # (2) .to('cuda') on tensors/modules (permanent; superset of the hub-load context mgr)
    def _tensor_to(self, *a, **k):
        if a and _is_cuda(a[0]):
            if self.dtype == torch.float64:
                self = self.float()
            a = (dev,) + tuple(a[1:])
        if _is_cuda(k.get("device")):
            if self.dtype == torch.float64:
                self = self.float()
            k = dict(k); k["device"] = dev
        return _orig_tensor_to(self, *a, **k)

    def _module_to(self, *a, **k):
        if a and _is_cuda(a[0]):
            a = (dev,) + tuple(a[1:])
        if _is_cuda(k.get("device")):
            k = dict(k); k["device"] = dev
        return _orig_module_to(self, *a, **k)

    torch.Tensor.to = _tensor_to
    torch.nn.Module.to = _module_to

    # NOTE: we deliberately do NOT wrap torch factory functions (torch.tensor/zeros/...)
    # to remap device='cuda'. Doing so breaks torch.jit.script (geffnet/DSINE scripts
    # functions that call torch.tensor; TorchScript can't compile a *args/**kwargs
    # Python wrapper). Hardcoded device='cuda' in factory calls is handled at the call
    # site (see anigen.utils.postprocessing_utils._active_device).


configure_mps_environment()
install_nvdiffrast_alias()
install_knn_shim()
install_cuda_redirect()
