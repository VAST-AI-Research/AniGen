import importlib

__attributes = {
    'BasicTrainer': 'basic',

    'AniGenSparseStructureVaeTrainer': 'vae.anigen_sparse_structure_vae',
    'AniGenSLatVaeSkeletonTrainer': 'vae.anigen_slat_mesh_vae',
    'AniGenSLatGaussianVAETrainer': 'vae.anigen_slat_gs_vae',
    'AniGenSkinAETrainer': 'vae.anigen_skin_ae',
    'AniGenFlowMatchingTrainer': 'flow_matching.anigen_flow_matching',
    'AniGenFlowMatchingCFGTrainer': 'flow_matching.anigen_flow_matching',
    'AniGenTextConditionedFlowMatchingCFGTrainer': 'flow_matching.anigen_flow_matching',
    'AniGenImageConditionedFlowMatchingCFGTrainer': 'flow_matching.anigen_flow_matching',   
    'AniGenSparseFlowMatchingTrainer': 'flow_matching.anigen_sparse_flow_matching',
    'AniGenSparseFlowMatchingCFGTrainer': 'flow_matching.anigen_sparse_flow_matching',
    'AniGenTextConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.anigen_sparse_flow_matching',
    'AniGenImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.anigen_sparse_flow_matching',
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
