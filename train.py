import os
import sys
import json
import glob
import argparse
import traceback
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random

from anigen import models, datasets, trainers
from anigen.utils.dist_utils import setup_dist
import torch.distributed as dist

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:  # pragma: no cover - elastic may be unavailable in older Torch versions
    def record(fn):  # type: ignore
        return fn


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', '*.pt'))
            if len(files) != 0:
                steps = []
                for f in files:
                    try:
                        steps.append(int(os.path.basename(f).split('step')[-1].split('.')[0]))
                    except (ValueError, IndexError):
                        pass
                if len(steps) > 0:
                    cfg.load_ckpt = max(steps)
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    # if world_size > 1:
    setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args, local_rank=local_rank)

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            # print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt, skip_init_snapshot=cfg.skip_init_snapshot)

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()
    
    if dist.is_initialized():
        dist.destroy_process_group()


@record
def _spawn_worker(local_rank, cfg):
    try:
        main(local_rank, cfg)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=3, help='Number of retries on error')
    ## dubug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    parser.add_argument('--skip_init_snapshot', action='store_true', help='Skip initial snapshot of dataset')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    config = json.load(open(opt.config, 'r'))
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
 
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('TORCH_DISABLE_ADDR2LINE', '1')
    os.environ.setdefault('PYTHONFAULTHANDLER', '1')
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(_spawn_worker, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            _spawn_worker(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(_spawn_worker, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    _spawn_worker(0, cfg)
                break
            except Exception as e:
                traceback.print_exc()
                print(f'Error: {e}', flush=True)
                if rty + 1 == cfg.auto_retry:
                    raise
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...', flush=True)
            