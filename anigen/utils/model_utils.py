import os
import json
import torch
from easydict import EasyDict as edict
from anigen import models

def load_model_from_path(path, model_name_in_config=None, device='cuda', use_ema=False):
    if os.path.isdir(path):
        config_path = os.path.join(path, 'config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found in {path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        config = edict(config)
        
        ckpt_dir = os.path.join(path, 'ckpts')
        if not os.path.exists(ckpt_dir):
             raise ValueError(f"Checkpoints directory not found in {path}")
        
        files = os.listdir(ckpt_dir)
        pt_files = [f for f in files if f.endswith('.pt')]
        if not pt_files:
            raise ValueError(f"No .pt files found in {ckpt_dir}")
        
        def get_step(name):
            try:
                return int(name.split('step')[-1].split('.')[0])
            except:
                return -1
        
        # Filter for EMA if requested
        if use_ema:
            ema_files = [f for f in pt_files if 'ema' in f]
            if ema_files:
                pt_files = ema_files
                print("Selected EMA checkpoint.")
            else:
                print("Warning: EMA checkpoint requested but not found. Falling back to regular checkpoint.")
                pt_files = [f for f in pt_files if 'ema' not in f and 'misc' not in f]
        else:
            # Exclude 'misc' checkpoints which contain optimizer state, not model weights
            non_ema_files = [f for f in pt_files if 'ema' not in f and 'misc' not in f]
            if non_ema_files:
                pt_files = non_ema_files
                print("Selected regular checkpoint.")
            else:
                print("Warning: Regular checkpoint not found. Falling back to EMA checkpoint.")
                pt_files = [f for f in pt_files if 'ema' in f]
        
        pt_files.sort(key=get_step, reverse=True)
        ckpt_path = os.path.join(ckpt_dir, pt_files[0])
        print(f"Loading checkpoint: {ckpt_path}")
        
        if model_name_in_config:
            model_config = config.models[model_name_in_config]
        else:
            keys = list(config.models.keys())
            # Heuristic: prefer 'denoiser' or 'flow_model'
            if 'denoiser' in keys:
                model_config = config.models['denoiser']
            elif len(keys) == 1:
                model_config = config.models[keys[0]]
            else:
                raise ValueError(f"Multiple models in config {keys}, please specify model_name_in_config")

        model = getattr(models, model_config.name)(**model_config.args)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, config
    else:
        raise ValueError("Please provide a directory containing config.json and ckpts/")

def load_decoder(path, ckpt_name, device):
    if not os.path.exists(path):
        raise ValueError(f"Decoder path not found: {path}")
    
    config_path = os.path.join(path, 'config.json')
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found in {path}")
        
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    if 'models' not in cfg or 'decoder' not in cfg['models']:
        raise ValueError(f"Config at {path} does not have ['models']['decoder']")
        
    model_cfg = cfg['models']['decoder']
    decoder = getattr(models, model_cfg['name'])(**model_cfg['args'])
    
    ckpt_path = os.path.join(path, 'ckpts', f'decoder_{ckpt_name}.pt')
    if not os.path.exists(ckpt_path):
        # Fallback to just ckpt_name if decoder_ prefix not found
        ckpt_path = os.path.join(path, 'ckpts', f'{ckpt_name}.pt')
        if not os.path.exists(ckpt_path):
             raise ValueError(f"Checkpoint not found: {ckpt_path}")
            
    print(f"Loading decoder from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    decoder.load_state_dict(state_dict, strict=False)
    decoder.to(device).eval()
    return decoder
