import importlib

__attributes = {
    'AniGenSparseStructureEncoder': 'anigen_sparse_structure_vae',
    'AniGenSparseStructureDecoder': 'anigen_sparse_structure_vae',
    'AniGenSparseStructureFlowModel': 'anigen_sparse_structure_flow',
    'AniGenSparseStructureFlowModelInpaint': 'anigen_sparse_structure_flow_inpaint',
    'AniGenElasticSLatEncoder': 'structured_latent_vae',
    'AniGenElasticSLatMeshDecoder': 'structured_latent_vae',
    'AniGenElasticSLatGaussianDecoder': 'structured_latent_vae',
    'AniGenSLatFlowModel': 'anigen_structured_latent_flow',
    'AniGenElasticSLatFlowModel': 'anigen_structured_latent_flow',
    'AniGenElasticSLatFlowModelOld': 'anigen_structured_latent_flow_old',
    'SkinAutoEncoder': 'structured_latent_vae',
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


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        print(f"{path}.json and {path}.safetensors not found, trying to download from Hugging Face Hub.")
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file))

    return model

