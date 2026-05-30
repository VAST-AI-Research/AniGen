import platform
if platform.system() == 'Darwin':
    from .metal_backend import MtlMesh as CuMesh
    from mtlbvh import MtlBVH as cuBVH
    from . import metal_remeshing as remeshing
else:
    from .cumesh import CuMesh
    from .bvh import cuBVH
    from . import remeshing
from .xatlas import Atlas
