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
        if hasattr(model, "use_fp16"):
            model.use_fp16 = False
        if hasattr(model, "dtype"):
            model.dtype = torch.float32


configure_mps_environment()
install_nvdiffrast_alias()
install_knn_shim()
