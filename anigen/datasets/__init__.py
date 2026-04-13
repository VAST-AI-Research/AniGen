import importlib

__attributes = {
    'AniGenSparseStructure': 'anigen_sparse_structure',
    'AniGenSparseFeat2Skeleton': 'anigen_sparse_feat2skeleton',
    'AniGenSparseFeat2Render': 'anigen_sparse_feat2render',
    
    'AniGenSparseStructureLatent': 'anigen_sparse_structure_latent',
    'TextConditionedAniGenSparseStructureLatent': 'anigen_sparse_structure_latent',
    'ImageConditionedAniGenSparseStructureLatent': 'anigen_sparse_structure_latent',

    'AniGenSLat': 'anigen_structured_latent',
    'AniGenTextConditionedSLat': 'anigen_structured_latent',
    'AniGenImageConditionedSLat': 'anigen_structured_latent',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]
