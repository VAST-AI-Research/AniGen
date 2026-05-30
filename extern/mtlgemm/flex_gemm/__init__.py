import os
USE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_CACHE', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE', '1') == '1'
AUTOTUNE_CACHE_PATH = os.environ.get(
    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
    os.path.expanduser('~/.flex_gemm/autotune_cache.json')
)
    

from . import kernels
from . import ops
from . import utils


if USE_AUTOTUNE_CACHE:
    utils.load_autotune_cache()
