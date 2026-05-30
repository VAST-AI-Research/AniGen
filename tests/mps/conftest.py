"""Test bootstrap for the Apple Silicon (MPS) inference port.

Importing anigen_mps first sets the backend env vars (SPARSE_ATTN_BACKEND=naive,
etc.) BEFORE anigen.modules.sparse reads them at import time.

`anigen/__init__.py` eagerly imports the full pipeline stack, which transitively
pulls in heavy native deps that are ported in *later* tasks (spconv / flex_gemm
sparse conv, nvdiffrast / pytorch3d rendering, rembg preprocessing). None of
these are needed to exercise the sparse-ATTENTION fp32 fallback. To let the
package import on a clean macOS env we register lightweight stand-in modules for
any of them that are not installed. This is purely test scaffolding and becomes a
no-op once the real backends are available.
"""
import sys
import types
import importlib.util
import importlib.abc

import anigen_mps  # noqa: F401  -- sets backend env vars at import time

# Third-party deps referenced (often eagerly, at import time) by anigen/* whose
# real implementations are delivered by OTHER tasks of the port (sparse conv,
# rendering, training, pipeline I/O) or are simply not needed to exercise the
# sparse-attention fp32 fallback. Any of these that are not installed get a lazy
# stub so `import anigen.modules.sparse...` can complete on a clean macOS env.
_OPTIONAL_NATIVE_DEPS = (
    "spconv", "torchsparse",          # sparse conv (Task 8/9)
    "nvdiffrast", "diff_gaussian_rasterization", "diffoctreerast",
    # NOTE: pytorch3d is intentionally NOT stubbed here. anigen_mps.install_knn_shim()
    # (run on `import anigen_mps` above) provides a real pytorch3d.ops with CPU
    # cKDTree knn_points/ball_query drop-ins (Task 9). Stubbing it would clobber
    # those with no-op stand-ins.
    "kaolin", "open3d", "vox2seq",  # rendering / 3D
    "rembg",                          # image preprocessing
    "flash_attn", "xformers",         # other attention backends (naive is active)
    "cubvh",                          # training-only CUDA ext
    "lpips", "muon",                  # training losses / optimizer
    "pandas", "plyfile", "pygltflib", "sklearn",  # misc pipeline/data utils
)


class _LazyStubModule(types.ModuleType):
    """A module whose every attribute / submodule resolves to another stub.

    Enough to satisfy import-time `from x.y import Z` / `import x.y` without the
    real package being installed. Calling a resolved attribute returns another
    stub so trivial import-time usage does not explode.
    """

    def __getattr__(self, name):  # noqa: D401
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child = _LazyStubModule(f"{self.__name__}.{name}")
        sys.modules[child.__name__] = child
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return _LazyStubModule(f"{self.__name__}.<call>")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve any import under a stubbed top-level package to a lazy stub."""

    def __init__(self, roots):
        self._roots = tuple(roots)

    def _is_stubbed(self, fullname: str) -> bool:
        top = fullname.split(".", 1)[0]
        return top in self._roots

    def find_spec(self, fullname, path=None, target=None):
        if not self._is_stubbed(fullname):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        mod = _LazyStubModule(spec.name)
        mod.__path__ = []  # mark as a package so submodule imports proceed
        return mod

    def exec_module(self, module):
        return None


def _install_stub_finder() -> None:
    missing = [m for m in _OPTIONAL_NATIVE_DEPS if importlib.util.find_spec(m) is None]
    if not missing:
        return
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder(missing))


_install_stub_finder()
