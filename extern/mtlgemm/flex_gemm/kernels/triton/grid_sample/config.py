import triton
from ..utils import get_autotune_config


autotune_config = get_autotune_config(
    default=[
        triton.Config({'BM': 4,   'BK': 64}, num_warps=2),
        triton.Config({'BM': 2,   'BK': 64}, num_warps=2),
        triton.Config({'BM': 1,   'BK': 64}, num_warps=2),
        triton.Config({'BM': 8,   'BK': 32}, num_warps=2),
        triton.Config({'BM': 4,   'BK': 32}, num_warps=2),
        triton.Config({'BM': 2,   'BK': 32}, num_warps=2),
        triton.Config({'BM': 16,  'BK': 16}, num_warps=2),
        triton.Config({'BM': 8,   'BK': 16}, num_warps=2),
        triton.Config({'BM': 4,   'BK': 16}, num_warps=2),
        triton.Config({'BM': 32,  'BK': 8 }, num_warps=2),
        triton.Config({'BM': 16,  'BK': 8 }, num_warps=2),
        triton.Config({'BM': 8,   'BK': 8 }, num_warps=2),
    ]
)
