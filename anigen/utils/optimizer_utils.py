"""
Optimizer utilities for PyTorch Lightning systems.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any


def create_muon_optimizer(models: Dict[str, nn.Module], optimizer_args: Dict[str, Any]):
    """
    Create Muon optimizer with different learning rates for different parameters.
    
    Args:
        models: Dictionary of models
        optimizer_args: Optimizer configuration with muon_lr, muon_weight_decay, 
                       other_lr, other_weight_decay
    
    Returns:
        Configured optimizer
    """
    muon_lr = optimizer_args.get('muon_lr', 0.02)
    muon_weight_decay = optimizer_args.get('muon_weight_decay', 0.01)
    other_lr = optimizer_args.get('other_lr', 1e-4)
    other_weight_decay = optimizer_args.get('other_weight_decay', 0.01)
    
    # Separate parameters for Muon and other optimizers
    muon_params = []
    other_params = []
    
    for name, model in models.items():
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Define which parameters should use Muon optimizer
            # This is a heuristic - adjust based on your model architecture
            if ('weight' in param_name and 
                param.dim() >= 2 and 
                param.numel() >= 1024):  # Large weight matrices
                muon_params.append(param)
            else:
                other_params.append(param)
    
    # Try to import and use Muon optimizer
    try:
        from muon import Muon
        
        # Create parameter groups
        param_groups = []
        
        if muon_params:
            param_groups.append({
                'params': muon_params,
                'lr': muon_lr,
                'weight_decay': muon_weight_decay,
                'momentum': 0.95,  # Muon-specific parameter
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': other_lr,
                'weight_decay': other_weight_decay,
            })
        
        return Muon(param_groups)
        
    except ImportError:
        print("Warning: Muon optimizer not available. Falling back to AdamW.")
        # Fallback to AdamW with combined parameters
        all_params = muon_params + other_params
        return torch.optim.AdamW(all_params, lr=other_lr, weight_decay=other_weight_decay)


def create_optimizer(models: Dict[str, nn.Module], optimizer_config: Dict[str, Any]):
    """
    Create optimizer based on configuration.
    
    Args:
        models: Dictionary of models
        optimizer_config: Optimizer configuration
    
    Returns:
        Configured optimizer
    """
    optimizer_name = optimizer_config.get('name', 'Adam').lower()
    optimizer_args = optimizer_config.get('args', {})
    
    # Get all parameters
    params = []
    for model in models.values():
        params.extend([p for p in model.parameters() if p.requires_grad])
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(params, **optimizer_args)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(params, **optimizer_args)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(params, **optimizer_args)
    elif optimizer_name == 'muon':
        return create_muon_optimizer(models, optimizer_args)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_lr_scheduler(optimizer, lr_scheduler_config: Dict[str, Any]):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        lr_scheduler_config: Scheduler configuration
    
    Returns:
        Configured scheduler
    """
    scheduler_name = lr_scheduler_config.get('name', 'CosineAnnealingLR')
    scheduler_args = lr_scheduler_config.get('args', {})
    
    if scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
    elif scheduler_name == 'LinearLR':
        return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_args)
    elif scheduler_name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
    elif scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
    elif scheduler_name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
    elif scheduler_name == 'SequentialLR':
        # Handle sequential scheduler
        schedulers = []
        for sched_config in lr_scheduler_config['schedulers']:
            sched = create_single_scheduler(sched_config, optimizer)
            schedulers.append(sched)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, **scheduler_args
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def create_single_scheduler(scheduler_config: Dict[str, Any], optimizer):
    """Create a single scheduler for SequentialLR."""
    name = scheduler_config['name']
    args = scheduler_config['args']
    
    if name == 'LinearLR':
        return torch.optim.lr_scheduler.LinearLR(optimizer, **args)
    elif name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **args)
    elif name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **args)
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **args)
    elif name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **args)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


class AdaptiveGradClipper:
    """
    Adaptive gradient clipping based on percentile of gradient norms.
    """
    
    def __init__(self, max_norm: float = 1.0, clip_percentile: float = 95):
        self.max_norm = max_norm
        self.clip_percentile = clip_percentile
        self.grad_norms_history = []
        self.history_size = 1000
    
    def __call__(self, parameters):
        """Apply adaptive gradient clipping."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        
        if not parameters:
            return torch.tensor(0.0)
        
        # Calculate gradient norm
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in parameters])
        )
        
        # Update history
        self.grad_norms_history.append(total_norm.item())
        if len(self.grad_norms_history) > self.history_size:
            self.grad_norms_history.pop(0)
        
        # Calculate adaptive threshold
        if len(self.grad_norms_history) >= 10:
            threshold = torch.quantile(
                torch.tensor(self.grad_norms_history), 
                self.clip_percentile / 100.0
            )
            clip_value = min(self.max_norm, threshold.item())
        else:
            clip_value = self.max_norm
        
        # Apply clipping
        if total_norm > clip_value:
            clip_coef = clip_value / (total_norm + 1e-6)
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm


def create_grad_clipper(grad_clip_config: Dict[str, Any]):
    """Create gradient clipper based on configuration."""
    if grad_clip_config is None:
        return None
    
    name = grad_clip_config.get('name', 'norm')
    args = grad_clip_config.get('args', {})
    
    if name.lower() == 'norm':
        max_norm = args.get('max_norm', 1.0)
        return lambda params: torch.nn.utils.clip_grad_norm_(params, max_norm)
    elif name.lower() == 'adaptivegradclipper':
        return AdaptiveGradClipper(**args)
    else:
        raise ValueError(f"Unknown gradient clipper: {name}")
