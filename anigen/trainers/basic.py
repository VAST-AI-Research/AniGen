from email.policy import strict
import os
import copy
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .utils import *
from .base import Trainer
from ..utils.general_utils import *
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils


class BasicTrainer(Trainer):
    """
    Trainer for basic training loop.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - FP16 mode: {self.fp16_mode}')
        return '\n'.join(lines)
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        if self.optimizer_config['name'].lower() == 'muon':
            self._ensure_uniform_param_dtype_for_muon()

        if self.world_size > 1:
            # Prepare distributed data parallel.
            # NOTE: PyTorch will raise if a module has no trainable parameters.
            self.training_models = {}
            for name, model in self.models.items():
                has_trainable_params = any(p.requires_grad for p in model.parameters())
                if has_trainable_params:
                    self.training_models[name] = DDP(
                        model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        bucket_cap_mb=128,
                        find_unused_parameters=False
                    )
                else:
                    self.training_models[name] = model
                    if self.is_master:
                        print(f"\n\033[93mWarning: Model '{name}' has no trainable parameters; skipping DDP wrapping.\033[0m")
        else:
            self.training_models = self.models

        if self.optimizer_config['name'].lower() == 'muon' and self.fp16_mode == 'inflat_all':
            self.fp16_mode = 'amp_manual'
            print(f'\n\033[93mWarning: Muon optimizer does not support inflat_all mode, switching to amp mode.\033[0m')
        
        # Build master params
        # self.model_params = sum([[p for p in model.parameters() if p.requires_grad] for model in self.models.values()], [])
        self.model_params_names, self.model_params = [], []
        for key in self.models:
            model = self.models[key]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.model_params_names.append(key + '.' + name)
                    self.model_params.append(param)

        if 'amp' in self.fp16_mode:
            self.master_params = self.model_params
            self.scaler = torch.GradScaler()
        elif self.fp16_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            self.fp16_scale_growth = self.fp16_scale_growth
            self.log_scale = 15.0
        elif self.loss_scaling:
            self.master_params = make_master_params_regular(self.model_params)
            self.log_scale = 15.0
        else:
            self.master_params = self.model_params

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if self.optimizer_config['name'].lower() == 'muon':
            from muon import MuonWithAuxAdam
            hidden_attn_params, other_params = [], []
            for n, p in zip(self.model_params_names, self.master_params):
                attn_only = self.optimizer_config['args'].get('attn_only', False)
                if ((attn_only and 'attn' in n) or not attn_only) and p.ndim >= 2:
                    hidden_attn_params.append(p)
                else:
                    other_params.append(p)
                param_groups = [
                    dict(use_muon=True, params=hidden_attn_params, lr=self.optimizer_config['args']['muon_lr'], weight_decay=self.optimizer_config['args'].get('muon_weight_decay', 0.01)),
                    dict(use_muon=False, params=other_params, lr=self.optimizer_config['args']['other_lr'], weight_decay=self.optimizer_config['args'].get('other_weight_decay', 0.01))
                ]
            self.optimizer = MuonWithAuxAdam(param_groups)
        elif hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                if self.lr_scheduler_config['name'] == 'SequentialLR':
                    scheduler_list = []
                    for scheduler_config in self.lr_scheduler_config['schedulers']:
                        scheduler_list.append(getattr(torch.optim.lr_scheduler, scheduler_config['name'])(self.optimizer, **scheduler_config['args']))
                    self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, scheduler_list, **self.lr_scheduler_config['args'])
                else:
                    self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def _ensure_uniform_param_dtype_for_muon(self):
        """Muon optimizer gathers parameters across ranks and requires uniform dtypes."""
        target_dtype = self._determine_muon_param_dtype()
        casted_any = False
        for model in self.models.values():
            if any(param.dtype != target_dtype for param in model.parameters() if param.requires_grad):
                casted_any = True
                if not self._cast_model_to_dtype(model, target_dtype):
                    for param in model.parameters():
                        if param.requires_grad and param.dtype != target_dtype:
                            param.data = param.data.to(dtype=target_dtype)
                    for buffer in model.buffers():
                        if buffer.is_floating_point() and buffer.dtype != target_dtype:
                            buffer.data = buffer.data.to(dtype=target_dtype)
        if casted_any and self.is_master:
            print(f"\nMuon optimizer converted trainable parameters to {target_dtype} to keep dtype consistent across ranks.")

    def _determine_muon_param_dtype(self):
        if isinstance(self.muon_param_dtype, torch.dtype):
            return self.muon_param_dtype
        if self.muon_param_dtype is None:
            return torch.float32
        key = str(self.muon_param_dtype).lower()
        if key in ('fp16', 'float16', 'half'):
            return torch.float16
        if key in ('bf16', 'bfloat16'):
            return torch.bfloat16
        if key in ('fp32', 'float32'):
            return torch.float32
        raise ValueError(f'Unsupported muon_param_dtype: {self.muon_param_dtype}')

    def _cast_model_to_dtype(self, model, target_dtype):
        if target_dtype == torch.float16 and hasattr(model, 'convert_to_fp16'):
            model.convert_to_fp16()
            return True
        if target_dtype == torch.float32 and hasattr(model, 'convert_to_fp32'):
            model.convert_to_fp32()
            return True
        if target_dtype == torch.bfloat16 and hasattr(model, 'convert_to_bf16'):
            model.convert_to_bf16()
            return True
        return False

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.fp16_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts, strict=True):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()], [])
        if strict:
            params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        else:
            param_dict = {key: dict(value.named_parameters()) for key, value in self.models.items()}
            params = []
            for i, (name, param_name) in enumerate(master_params_names):
                if name in state_dicts and param_name in state_dicts[name]:
                    params.append(state_dicts[name][param_name].to(master_params[0].device))
                else:
                    params.append(param_dict[name][param_name].data.clone().to(master_params[0].device))
        if self.fp16_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load_pretrain(self, load_dir):
        """
        Load pretrained checkpoints.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading pretrained checkpoint from step {load_dir}...', end='')

        need_load = False
        model_ckpts = {}
        for name, model in self.models.items():
            if hasattr(model, 'use_pretrain_branch') and model.use_pretrain_branch:
                class_names = model.__class__.__name__ if not hasattr(model, 'pretrain_class_name') else model.pretrain_class_name
                need_load = True
                from safetensors.torch import load_file
                if type(class_names) is not list:
                    class_names = [class_names]
                for class_name in class_names:
                    if os.path.exists(os.path.join('ckpts/pretrained_ckpts', f'{class_name}.safetensors')):
                        pretrain_ckpt = load_file(os.path.join('ckpts/pretrained_ckpts', f'{class_name}.safetensors'))
                    elif os.path.exists(os.path.join('ckpts/pretrained_ckpts', f'{class_name}.pt')):
                        pretrain_ckpt = torch.load(os.path.join('ckpts/pretrained_ckpts', f'{class_name}.pt'), weights_only=True)
                    else:
                        continue
                    if hasattr(model, 'pretrain_ckpt_filter_prefix') and class_name in model.pretrain_ckpt_filter_prefix:
                        prefix = model.pretrain_ckpt_filter_prefix[class_name]
                        pretrain_ckpt = {key: value for key, value in pretrain_ckpt.items() if key.startswith(prefix)}
                    model_ckpts[name] = pretrain_ckpt
                    model.load_state_dict(pretrain_ckpt, strict=False)
        if need_load:
            self._state_dicts_to_master_params(self.master_params, model_ckpts, strict=False)
        del model_ckpts
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Load Pretrained Checkpoints Done.')
        if self.world_size > 1:
            self.check_ddp()

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            if os.path.exists(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')):
                model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt, strict=True)
            else:
                model_ckpts[name] = model.state_dict()
            if self.fp16_mode == 'inflat_all' or self.loss_scaling:
                model.convert_to_fp16()
        self._state_dicts_to_master_params(self.master_params, model_ckpts, strict=True)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ckpt_path = os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt')
                    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')  # If EMA ckpt not found, load normal ckpt instead
                    if os.path.exists(ckpt_path):
                        ema_ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                        ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts, strict=False)
                del ema_ckpts
        misc_ckpt_path = os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')
        if os.path.exists(misc_ckpt_path):
            misc_ckpt = torch.load(read_file_dist(misc_ckpt_path), map_location=torch.device('cpu'), weights_only=False)
            self.optimizer.load_state_dict(misc_ckpt['optimizer'])
            self.step = misc_ckpt['step']
            self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
            if 'amp' in self.fp16_mode:
                self.scaler.load_state_dict(misc_ckpt['scaler'])
            elif self.fp16_mode == 'inflat_all' or self.loss_scaling:
                self.log_scale = misc_ckpt['log_scale']
            if self.lr_scheduler_config is not None:
                self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
            if self.elastic_controller_config is not None:
                self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
            if self.grad_clip is not None and not isinstance(self.grad_clip, float):
                self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
            del misc_ckpt
        else:
            self.step = step

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if 'amp' in self.fp16_mode:
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.fp16_mode == 'inflat_all' or self.loss_scaling:
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

        if not hasattr(self, 'model_saved_steps'):
            self.model_saved_steps = [self.step]
        else:
            self.model_saved_steps.append(self.step)
        
        # Deleting old checkpoints and keep the last 5 checkpoints
        self.manage_checkpoints()

    def manage_checkpoints(self):
        def get_step_from_filename(filename, prefix, suffix):
            match = re.search(rf'{prefix}(\d+){suffix}', filename)
            return int(match.group(1)) if match else None
        ckpt_dir = os.path.join(self.output_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            print("Checkpoint directory does not exist, skipping cleanup.")
            return
        checkpoints = []
        total_steps = []
        for filename in os.listdir(ckpt_dir):
            if filename.endswith('.pt'):
                step = get_step_from_filename(filename, 'step', '.pt')
                if step is not None:
                    checkpoints.append((step, filename))
                    total_steps.append(step)
        checkpoints.sort(key=lambda x: x[0])
        total_steps = sorted(np.unique(total_steps).tolist())
        # Avoid deleting steps mod 100000 to 0
        total_steps = [step for step in total_steps if step % 100000 != 0]

        # Keep only the latest 2 checkpoints
        if len(total_steps) > 2:
            old_steps = total_steps[:-2]
            old_checkpoints = [(step, filename) for step, filename in checkpoints if step in old_steps]
            for step, filename in old_checkpoints:
                os.remove(os.path.join(ckpt_dir, filename))
                print(f'\nDeleted old checkpoint file: {filename}')
            for old_step, _ in old_checkpoints:
                for name in self.models:
                    model_ckpt = os.path.join(ckpt_dir, f'{name}_step{old_step:07d}.pt')
                    if os.path.exists(model_ckpt):
                        os.remove(model_ckpt)
                        print(f'Deleted model checkpoint: {model_ckpt}')
                    for ema_rate in self.ema_rate:
                        ema_ckpt = os.path.join(ckpt_dir, f'{name}_ema{ema_rate}_step{old_step:07d}.pt')
                        if os.path.exists(ema_ckpt):
                            os.remove(ema_ckpt)
                            print(f'Deleted EMA checkpoint: {ema_ckpt}')
                misc_ckpt = os.path.join(ckpt_dir, f'misc_step{old_step:07d}.pt')
                if os.path.exists(misc_ckpt):
                    os.remove(misc_ckpt)
                    print(f'Deleted misc checkpoint: {misc_ckpt}')

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=True)
                for k, v in model_ckpt.items():
                    if model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
                if self.fp16_mode == 'inflat_all' or self.loss_scaling:
                    model.convert_to_fp16()
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for n, p in zip(self.model_params_names, self.master_params):
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().reshape(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {'loss': {}, 'status': {}}
        amp_context = partial(torch.autocast, device_type='cuda') if self.fp16_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [getattr(self.training_models[name], 'no_sync', nullcontext) for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if 'amp' in self.fp16_mode:
                    self.scaler.scale(l).backward()
                elif self.fp16_mode == 'inflat_all':
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                elif self.loss_scaling:
                    # Mirror manual loss scaling path to keep updates unbiased.
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())

        ## gradient clip
        if self.grad_clip is not None:
            if 'amp' in self.fp16_mode:
                self.scaler.unscale_(self.optimizer)
            elif self.fp16_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                scale = 2 ** self.log_scale
                self.master_params[0].grad.mul_(1.0 / scale if scale > 1e-30 else 1.0)
            elif self.loss_scaling:
                model_grads_to_master_grads_regular(self.model_params, self.master_params)
                scale = 2 ** self.log_scale
                inv_scale = 1.0 / scale if scale > 1e-30 else 1.0
                for p in self.master_params:
                    if p.grad is not None:
                        p.grad.mul_(inv_scale)
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()

        # Cache params for NaN recovery
        if self.step % 10 == 0 and not any(not p.isfinite().all() for p in self.master_params):
            self.previous_step = self.step
            self.previous_params = [p.clone().detach() for p in self.master_params]
            self.previous_params_model = [p.clone().detach() for p in self.model_params]

        # # Check which parameter does not have gradients and print its name
        # for name, param in self.training_models.items():
        #     for p_name, p in param.named_parameters():
        #         if p.requires_grad and p.grad is None:
        #             print(f'\n\033[93mWarning: {name}.{p_name} does not have gradients at step {self.step}.\033[0m')
        # exit(0)

        ## step
        if 'amp' in self.fp16_mode:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.fp16_mode == 'inflat_all' or self.loss_scaling:
            prev_scale = 2 ** self.log_scale
            if not any(not p.grad.isfinite().all() for p in self.master_params if p.grad is not None):
                if self.grad_clip is None:
                    if self.fp16_mode == 'inflat_all':
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        scale = 2 ** self.log_scale
                        self.master_params[0].grad.mul_(1.0 / scale if scale > 1e-30 else 1.0)
                    else:
                        model_grads_to_master_grads_regular(self.model_params, self.master_params)
                        scale = 2 ** self.log_scale
                        inv_scale = 1.0 / scale if scale > 1e-30 else 1.0
                        for p in self.master_params:
                            if p.grad is not None:
                                p.grad.mul_(inv_scale)
                self.optimizer.step()
                if self.fp16_mode == 'inflat_all':
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    master_params_to_model_params_regular(self.model_params, self.master_params)
                self.log_scale += self.fp16_scale_growth
            else:
                self.log_scale -= 1
                if self.log_scale < -100:
                    self.log_scale = -100
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # If parameters are not finite, revert to previous parameters
        if any(not p.isfinite().all() for p in self.master_params):
            if hasattr(self, 'previous_params'):
                print(f'\n\033[91mError: NaN loss detected in parameters at step {self.step}. Reverting to previous parameters at step {self.previous_step}.\033[0m')
                for i, p in enumerate(self.master_params):
                    p.data.copy_(self.previous_params[i].data)
                for i, p in enumerate(self.model_params):
                    p.data.copy_(self.previous_params_model[i].data)
            else:
                print(f'\n\033[91mError: NaN loss detected in parameters at step {self.step}. No previous parameters to revert to.\033[0m')
                raise RuntimeError('NaN loss detected in parameters and no previous parameters to revert to.')

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            for key in self.models:
                param_norms = {}
                param_grads = {}
                for name, param in self.models[key].named_parameters():
                    if param.requires_grad:
                        param_norms[name] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[name] = param.grad.norm().item() / prev_scale
                step_log[key+'-param_norms'] = param_norms
                step_log[key+'-param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log
