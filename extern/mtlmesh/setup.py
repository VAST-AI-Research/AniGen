"""
Unified build script for cumesh.
- macOS (Apple Silicon): Metal compute shaders
- Linux/Windows (CUDA): CUDA kernels
- Override with BUILD_TARGET=cuda|metal|auto (default: auto)
"""
import os
import sys
import platform
import subprocess
import glob
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")


def _detect_backend():
    if BUILD_TARGET == "metal":
        return "metal"
    if BUILD_TARGET == "cuda" or BUILD_TARGET == "rocm":
        return BUILD_TARGET
    # auto-detect
    if IS_MACOS:
        return "metal"
    return "cuda"


BACKEND = _detect_backend()


# =============================================================================
# Metal backend
# =============================================================================
class MetalBuildExt(_build_ext):
    """Compiles .metal shaders → .metallib, then builds Obj-C++ extension."""

    def build_extensions(self):
        # Register .mm as valid source extension
        self.compiler.src_extensions.append(".mm")

        # Compile Metal shaders
        metal_dir = os.path.join(ROOT, "src", "metal")
        metal_files = sorted(glob.glob(os.path.join(metal_dir, "*.metal")))

        if metal_files:
            print(f"Compiling {len(metal_files)} Metal shader files...")
            air_files = []
            for mf in metal_files:
                air = mf.replace(".metal", ".air")
                subprocess.check_call([
                    "xcrun", "-sdk", "macosx", "metal",
                    "-c", mf, "-o", air,
                    "-std=metal4.0", "-O2",
                    "-D__HAVE_ATOMIC_ULONG__=1",
                    "-D__HAVE_ATOMIC_ULONG_MIN_MAX__=1",
                    "-I", metal_dir,
                ])
                air_files.append(air)

            metallib_path = os.path.join(ROOT, "src", "cumesh.metallib")
            subprocess.check_call([
                "xcrun", "-sdk", "macosx", "metallib",
                *air_files, "-o", metallib_path,
            ])
            for air in air_files:
                os.remove(air)

            # Install metallib alongside extension
            for ext in self.extensions:
                ext_path = self.get_ext_fullpath(ext.name)
                ext_dir = os.path.dirname(ext_path)
                os.makedirs(ext_dir, exist_ok=True)
                shutil.copy2(metallib_path, os.path.join(ext_dir, "cumesh.metallib"))

            # Also copy to source tree for editable installs
            src_pkg_dir = os.path.join(ROOT, "cumesh")
            src_dest = os.path.join(src_pkg_dir, "cumesh.metallib")
            if not os.path.exists(src_dest) or not os.path.samefile(metallib_path, src_dest):
                shutil.copy2(metallib_path, src_dest)

        _build_ext.build_extensions(self)


def _metal_extensions():
    from torch.utils.cpp_extension import include_paths, library_paths

    torch_includes = include_paths()
    torch_libs = library_paths()

    mtlmesh_ext = Extension(
        name="cumesh._C",
        sources=[
            "src/metal_context.mm",
            "src/metal_primitives.mm",
            "src/metal_hash.mm",
            "src/mtlmesh.mm",
            "src/ext.mm",
        ],
        include_dirs=[
            os.path.join(ROOT, "src"),
            os.path.join(ROOT, "src", "metal"),
        ] + torch_includes,
        library_dirs=torch_libs,
        extra_compile_args=[
            "-x", "objective-c++",
            "-std=c++17", "-O2", "-fobjc-arc",
            "-fmodules",
            "-DTORCH_EXTENSION_NAME=_C",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "Foundation",
            "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python",
        ],
        language="objc++",
    )

    ext_modules = [mtlmesh_ext]

    # xatlas (CPU only)
    xatlas_dir = os.path.join(ROOT, "third_party", "xatlas")
    if os.path.exists(xatlas_dir):
        ext_modules.append(Extension(
            name="cumesh._xatlas",
            sources=[
                os.path.join(xatlas_dir, "xatlas.cpp"),
                os.path.join(xatlas_dir, "binding.cpp"),
            ],
            include_dirs=torch_includes,
            library_dirs=torch_libs,
            extra_compile_args=["-O2", "-std=c++17",
                                "-DTORCH_EXTENSION_NAME=_xatlas",
                                "-DTORCH_API_INCLUDE_EXTENSION_H"],
            extra_link_args=["-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python"],
        ))

    return ext_modules


# =============================================================================
# CUDA backend
# =============================================================================
def _cuda_extensions():
    from torch.utils.cpp_extension import CUDAExtension, IS_HIP_EXTENSION

    IS_HIP = (BUILD_TARGET == "rocm") or (BUILD_TARGET == "auto" and bool(IS_HIP_EXTENSION))

    cxx_flags = []
    nvcc_flags = []
    if IS_WINDOWS:
        cxx_flags += ["/O2", "/std:c++17", "/EHsc", "/permissive-", "/Zc:__cplusplus"]
        nvcc_flags += ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--extended-lambda",
                       "-Xcompiler=/std:c++17", "-Xcompiler=/EHsc",
                       "-Xcompiler=/permissive-", "-Xcompiler=/Zc:__cplusplus"]
    else:
        cxx_flags += ["-O3", "-std=c++17"]
        nvcc_flags += ["-O3", "-std=c++17"]

    if IS_HIP:
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        nvcc_flags += [f"--offload-arch={arch}" for arch in archs]
    elif IS_WINDOWS:
        nvcc_flags += ["-allow-unsupported-compiler"]

    ext_modules = [
        CUDAExtension(
            name="cumesh._C",
            sources=[
                "src/hash/hash.cu",
                "src/atlas.cu", "src/clean_up.cu", "src/cumesh.cu",
                "src/connectivity.cu", "src/geometry.cu", "src/io.cu",
                "src/simplify.cu", "src/shared.cu",
                "src/remesh/simple_dual_contour.cu",
                "src/remesh/svox2vert.cu",
                "src/ext.cpp",
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
        CUDAExtension(
            name="cumesh._cubvh",
            sources=[
                "third_party/cubvh/src/bvh.cu",
                "third_party/cubvh/src/api_gpu.cu",
                "third_party/cubvh/src/bindings.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/cubvh/include"),
                os.path.join(ROOT, "third_party/cubvh/third_party/eigen"),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags + [
                "--extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ]},
        ),
        CUDAExtension(
            name="cumesh._cumesh_xatlas",
            sources=[
                "third_party/xatlas/xatlas.cpp",
                "third_party/xatlas/binding.cpp",
            ],
            extra_compile_args={"cxx": cxx_flags},
        ),
    ]
    return ext_modules


# =============================================================================
# Select backend and build
# =============================================================================
if BACKEND == "metal":
    pkg_name = "cumesh"
    packages = ["cumesh"]
    ext_modules = _metal_extensions()
    cmdclass = {"build_ext": MetalBuildExt}
    install_requires = ["torch>=2.0", "tqdm"]
else:
    from torch.utils.cpp_extension import BuildExtension
    pkg_name = "cumesh"
    packages = ["cumesh"]
    ext_modules = _cuda_extensions()
    cmdclass = {"build_ext": BuildExtension}
    install_requires = ["torch>=2.0"]

setup(
    name=pkg_name,
    version="0.1.0",
    packages=packages,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=install_requires,
)
