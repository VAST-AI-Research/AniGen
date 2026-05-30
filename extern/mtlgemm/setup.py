from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import platform
import shutil
import sys
import subprocess
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if platform.system() == "Darwin":
        BUILD_TARGET = "metal"
    else:
        try:
            from torch.utils.cpp_extension import IS_HIP_EXTENSION
            if IS_HIP_EXTENSION:
                BUILD_TARGET = "rocm"
            else:
                BUILD_TARGET = "cuda"
        except ImportError:
            BUILD_TARGET = "cuda"

if BUILD_TARGET == "metal":
    METAL_DIR = os.path.join(ROOT, "flex_gemm", "kernels", "metal")

    class MetalBuildExt(build_ext):
        def build_extensions(self):
            # Teach the compiler about .mm files (Objective-C++)
            self.compiler.src_extensions.append('.mm')
            original_spawn = self.compiler.spawn
            def patched_spawn(cmd, **kwargs):
                # Insert -x objective-c++ before .mm source files
                new_cmd = list(cmd)
                for i, arg in enumerate(new_cmd):
                    if arg.endswith('.mm') and i > 0 and new_cmd[i-1] == '-c':
                        new_cmd.insert(i, 'objective-c++')
                        new_cmd.insert(i, '-x')
                        break
                return original_spawn(new_cmd, **kwargs)
            self.compiler.spawn = patched_spawn

            # Step 1: Compile .metal -> .air
            metal_sources = glob.glob(os.path.join(METAL_DIR, "**", "*.metal"), recursive=True)
            air_files = []

            include_dirs = [
                os.path.join(METAL_DIR, "hash"),
                os.path.join(METAL_DIR, "serialize"),
                os.path.join(METAL_DIR, "common"),
                os.path.join(METAL_DIR, "grid_sample"),
                os.path.join(METAL_DIR, "spconv"),
                METAL_DIR,
            ]

            include_flags = []
            for d in include_dirs:
                include_flags.extend(["-I", d])

            build_temp = os.path.join(self.build_temp, "metal")
            os.makedirs(build_temp, exist_ok=True)

            # Header dependencies: any .h/.metal changes should invalidate .air.
            metal_headers = glob.glob(os.path.join(METAL_DIR, "**", "*.h"), recursive=True)

            def _mtime(path):
                return os.path.getmtime(path) if os.path.exists(path) else 0.0

            max_header_mtime = max((_mtime(h) for h in metal_headers), default=0.0)

            for src in metal_sources:
                air = os.path.join(build_temp, os.path.basename(src).replace(".metal", ".air"))
                src_m = _mtime(src)
                air_m = _mtime(air)
                needs = air_m == 0.0 or air_m < src_m or air_m < max_header_mtime
                if needs:
                    cmd = [
                        "xcrun", "-sdk", "macosx", "metal",
                        "-c", src, "-o", air,
                        "-std=metal4.0", "-O2",
                        "-D__HAVE_ATOMIC_ULONG__=1",
                        "-D__HAVE_ATOMIC_ULONG_MIN_MAX__=1",
                    ] + include_flags
                    print(f"  Compiling {os.path.basename(src)} -> {os.path.basename(air)}")
                    subprocess.check_call(cmd)
                air_files.append(air)

            # Step 2: Link .air -> flex_gemm.metallib, skip if up to date.
            metallib = os.path.join(build_temp, "flex_gemm.metallib")
            metallib_m = _mtime(metallib)
            max_air = max((_mtime(a) for a in air_files), default=0.0)
            if metallib_m == 0.0 or metallib_m < max_air:
                cmd = ["xcrun", "-sdk", "macosx", "metallib"] + air_files + ["-o", metallib]
                print(f"  Linking -> flex_gemm.metallib")
                subprocess.check_call(cmd)
            else:
                print(f"  flex_gemm.metallib up to date, skipping link")

            # Step 3: Resolve torch paths and build the C++ extension
            for ext in self.extensions:
                if hasattr(ext, '_resolve_torch'):
                    ext._resolve_torch()
            super().build_extensions()

            # Step 4: Install metallib alongside the .so (both build dir and source dir)
            for ext in self.extensions:
                ext_path = self.get_ext_fullpath(ext.name)
                ext_dir = os.path.dirname(ext_path)
                dest = os.path.join(ext_dir, "flex_gemm.metallib")
                shutil.copy2(metallib, dest)
                print(f"  Installed flex_gemm.metallib -> {dest}")

            # Also copy to source tree for editable installs
            src_metal_dir = os.path.join(ROOT, "flex_gemm", "kernels", "metal")
            src_dest = os.path.join(src_metal_dir, "flex_gemm.metallib")
            if not os.path.exists(src_dest) or not os.path.samefile(dest, src_dest):
                shutil.copy2(metallib, src_dest)
                print(f"  Installed flex_gemm.metallib -> {src_dest}")

    class LazyMetalExtension(Extension):
        """Extension that resolves torch include/library paths at build time, not import time."""
        def __init__(self, *args, **kwargs):
            self._torch_resolved = False
            super().__init__(*args, **kwargs)

        def _resolve_torch(self):
            if self._torch_resolved:
                return
            self._torch_resolved = True
            import torch.utils.cpp_extension as cpp_ext
            self.include_dirs.extend(cpp_ext.include_paths())
            lib_paths = cpp_ext.library_paths()
            self.library_dirs.extend(lib_paths)
            # Add torch lib dirs to rpath so libc10.dylib etc. can be found at runtime
            for p in lib_paths:
                self.extra_link_args.extend(["-Wl,-rpath," + p])

    METAL_REL = os.path.join("flex_gemm", "kernels", "metal")

    # Dependencies: all .metal and .h files under the metal dir. This teaches
    # `pip install -e .` to invalidate cached artifacts when shader source
    # changes — without it, editable installs skip build_ext entirely.
    metal_deps = (
        glob.glob(os.path.join(METAL_DIR, "**", "*.metal"), recursive=True)
        + glob.glob(os.path.join(METAL_DIR, "**", "*.h"), recursive=True)
    )

    ext = LazyMetalExtension(
        name="flex_gemm.kernels.metal._C",
        sources=[
            os.path.join(METAL_REL, "ext.mm"),
            os.path.join(METAL_REL, "common", "metal_context.mm"),
        ],
        depends=metal_deps,
        include_dirs=[
            METAL_DIR,
            os.path.join(METAL_DIR, "common"),
            os.path.join(METAL_DIR, "hash"),
            os.path.join(METAL_DIR, "serialize"),
            os.path.join(METAL_DIR, "grid_sample"),
            os.path.join(METAL_DIR, "spconv"),
        ],
        library_dirs=[],
        libraries=["c10", "torch", "torch_cpu", "torch_python"],
        extra_compile_args=[
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-fobjc-arc",
            "-DTORCH_EXTENSION_NAME=_C",
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "Foundation",
            "-Wl,-rpath,@loader_path",
        ],
        language="objc++",
    )

    setup(
        name="flex_gemm",
        packages=[
            "flex_gemm",
            "flex_gemm.utils",
            "flex_gemm.ops",
            "flex_gemm.ops.spconv",
            "flex_gemm.ops.grid_sample",
            "flex_gemm.kernels",
            "flex_gemm.kernels.metal",
        ],
        ext_modules=[ext],
        cmdclass={"build_ext": MetalBuildExt},
        install_requires=["torch"],
    )

    # Copy autotune cache
    os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)
    cache_src = os.path.join(ROOT, "autotune_cache.json")
    if os.path.exists(cache_src):
        shutil.copyfile(cache_src, os.path.expanduser("~/.flex_gemm/autotune_cache.json"))

else:
    # Original CUDA/ROCm build
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
    import torch

    IS_HIP = BUILD_TARGET == "rocm"

    if not IS_HIP:
        cc_flag = ["--use_fast_math", "-allow-unsupported-compiler"]
    else:
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        cc_flag = [f"--offload-arch={arch}" for arch in archs]

    if platform.system() == "Windows":
        extra_compile_args = {
            "cxx": ["/O2", "/std:c++17", "/EHsc", "/openmp", "/permissive-", "/Zc:__cplusplus"],
            "nvcc": ["-O3", "-std=c++17", "-Xcompiler=/std:c++17", "-Xcompiler=/EHsc", "-Xcompiler=/permissive-", "-Xcompiler=/Zc:__cplusplus"] + cc_flag,
        }
    else:
        cxx11_abi = "1" if torch.compiled_with_cxx11_abi() else "0"
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", "-fopenmp", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"],
            "nvcc": ["-O3", "-std=c++17"] + cc_flag,
        }

    setup(
        name="flex_gemm",
        packages=[
            "flex_gemm",
            "flex_gemm.utils",
            "flex_gemm.ops",
            "flex_gemm.ops.spconv",
            "flex_gemm.ops.grid_sample",
            "flex_gemm.kernels",
            "flex_gemm.kernels.triton",
            "flex_gemm.kernels.triton.spconv",
            "flex_gemm.kernels.triton.grid_sample",
        ],
        ext_modules=[
            CUDAExtension(
                name="flex_gemm.kernels.cuda",
                sources=[
                    "flex_gemm/kernels/cuda/hash/hash.cu",
                    "flex_gemm/kernels/cuda/serialize/api.cu",
                    "flex_gemm/kernels/cuda/grid_sample/grid_sample.cu",
                    "flex_gemm/kernels/cuda/spconv/neighbor_map.cu",
                    "flex_gemm/kernels/cuda/ext.cpp",
                ],
                extra_compile_args=extra_compile_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        install_requires=[
            'torch',
        ]
    )

    # copy cache to tmp dir
    os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)
    cache_src = os.path.join(ROOT, "autotune_cache.json")
    if os.path.exists(cache_src):
        shutil.copyfile(cache_src, os.path.expanduser('~/.flex_gemm/autotune_cache.json'))
