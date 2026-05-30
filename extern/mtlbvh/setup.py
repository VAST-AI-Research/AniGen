"""
Build script for mtlbvh — Metal BVH for distance queries and ray tracing.
Compiles Metal shaders and Obj-C++ / C++ PyTorch extension.
"""
import os
import subprocess
import glob
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

ROOT = os.path.dirname(os.path.abspath(__file__))


class MetalBuildExt(_build_ext):
    """Compiles .metal shaders -> .metallib, then builds Obj-C++/C++ extension."""

    def build_extensions(self):
        self.compiler.src_extensions += [".mm", ".cpp"]

        metal_dir = os.path.join(ROOT, "src", "metal")
        metal_files = sorted(glob.glob(os.path.join(metal_dir, "*.metal")))

        if metal_files:
            air_files = []
            for mf in metal_files:
                air = mf.replace(".metal", ".air")
                subprocess.check_call([
                    "xcrun", "-sdk", "macosx", "metal",
                    "-c", mf, "-o", air,
                    "-std=metal4.0", "-O2",
                    "-D__HAVE_ATOMIC_ULONG__=1",
                    "-D__HAVE_ATOMIC_ULONG_MIN_MAX__=1",
                    "-I", os.path.join(ROOT, "src"),
                ])
                air_files.append(air)

            metallib_path = os.path.join(ROOT, "src", "mtlbvh.metallib")
            subprocess.check_call([
                "xcrun", "-sdk", "macosx", "metallib",
                *air_files, "-o", metallib_path,
            ])
            for air in air_files:
                os.remove(air)

            for ext in self.extensions:
                ext_path = self.get_ext_fullpath(ext.name)
                ext_dir = os.path.dirname(ext_path)
                os.makedirs(ext_dir, exist_ok=True)
                shutil.copy2(metallib_path, os.path.join(ext_dir, "mtlbvh.metallib"))

            # Also copy to source tree for editable installs
            src_dest = os.path.join(ROOT, "mtlbvh", "mtlbvh.metallib")
            shutil.copy2(metallib_path, src_dest)

        _build_ext.build_extensions(self)


def _metal_extensions():
    from torch.utils.cpp_extension import include_paths, library_paths

    torch_includes = include_paths()
    torch_libs = library_paths()

    return [Extension(
        name="mtlbvh._C",
        sources=[
            "src/bvh_build.cpp",
            "src/metal_bvh.mm",
            "src/ext.mm",
        ],
        include_dirs=[os.path.join(ROOT, "src")] + torch_includes,
        library_dirs=torch_libs,
        extra_compile_args=[
            "-std=c++17", "-O2", "-fobjc-arc",
            "-DTORCH_EXTENSION_NAME=_C",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
            "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python",
        ],
        language="objc++",
    )]


setup(
    name="mtlbvh",
    version="0.1.0",
    description="Metal-accelerated BVH for distance queries and ray tracing on Apple Silicon",
    author="gusto",
    license="MIT",
    url="https://github.com/gusto/mtlbvh",
    packages=["mtlbvh"],
    ext_modules=_metal_extensions(),
    cmdclass={"build_ext": MetalBuildExt},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
)
