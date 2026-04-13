import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

if os.name == "nt":

	# find cl.exe
	def find_cl_path():
		import glob
		for executable in ["Program Files (x86)", "Program Files"]:
			for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
				paths = sorted(glob.glob(f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"), reverse=True)
				if paths:
					return paths[0]

	# If cl.exe is not on path, try to find it.
	if os.system("where cl.exe >nul 2>nul") != 0:
		cl_path = find_cl_path()
		if cl_path is None:
			raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
		os.environ["PATH"] += ";" + cl_path
	else:
		# cl.exe was found in PATH, so we can assume that the user is already in a developer command prompt
		# In this case, BuildExtensions requires the following environment variable to be set such that it
		# won't try to activate a developer command prompt a second time.
		os.environ["DISTUTILS_USE_SDK"] = "1"

cpp_standard = 17

base_nvcc_flags = [
	"-O3",
	f"-std=c++{cpp_standard}",
	"--extended-lambda",
	"--expt-relaxed-constexpr",
	# The following definitions must be undefined
	# since we need half-precision operation.
	"-U__CUDA_NO_HALF_OPERATORS__",
	"-U__CUDA_NO_HALF_CONVERSIONS__",
	"-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
	base_cflags = ["-O3", f"-std=c++{cpp_standard}"]
	base_nvcc_flags += [
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
	]
elif os.name == "nt":
	base_cflags = ["/O2", f"/std:c++{cpp_standard}"]

'''
Usage:
python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)
python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)
python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)
'''
setup(
	ext_modules=[
		CUDAExtension(
			name='_cubvh',
			sources=[os.path.join('src', f) for f in [
				'bvh.cu',
				'api_gpu.cu',
				'bindings.cpp',
			]],
			include_dirs=[
				os.path.join(_src_path, 'include'),
				os.path.join(_src_path, 'third_party', 'eigen'),
			],
			extra_compile_args={
				'cxx': base_cflags,
				'nvcc': base_nvcc_flags,
			}
		),
	],
	cmdclass={
		'build_ext': BuildExtension,
	},
)
