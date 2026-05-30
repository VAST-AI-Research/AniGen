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


configure_mps_environment()
install_knn_shim()
