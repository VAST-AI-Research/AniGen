import platform
import os

_BACKEND = os.environ.get('FLEX_GEMM_BACKEND', 'auto')

if _BACKEND == 'auto':
    if platform.system() == 'Darwin':
        _BACKEND = 'metal'
    else:
        _BACKEND = 'cuda'

if _BACKEND == 'metal':
    from . import metal
    # Expose metal as both 'cuda' and 'triton' so ops layer imports work unchanged
    cuda = metal
    triton = metal
elif _BACKEND == 'cuda':
    from . import cuda
    from . import triton
