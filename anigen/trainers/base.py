from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler


class Trainer:
    """
    Base class for training.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        i_print=500,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        skip_init_snapshot=False,
        loss_scaling=True,
        muon_param_dtype=None,
        num_workers_dl=None,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.num_workers_dl = num_workers_dl if num_workers_dl is not None else int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.skip_init_snapshot = skip_init_snapshot
        self.loss_scaling = loss_scaling
        self.muon_param_dtype = muon_param_dtype

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        if step is None or step == 0:
            self.load_pretrain(load_dir)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers_dl,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers_dl > 0 else False,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)
    
    def load_pretrain(self, load_dir):
        """
        Load pretrained checkpoints.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        elif vis is None:
            save_cfg = []
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False, force_no_split=False, **kwargs):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size)) if not force_no_split else num_samples
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose, **kwargs)

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in sorted(samples.keys()):
                samples[key]['value'] = samples[key]['value'].contiguous() if type(samples[key]['value']) is torch.Tensor else samples[key]['value']
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)] if type(samples[key]['value']) is torch.Tensor else [None for _ in range(self.world_size)]
                else:
                    all_images = []
                if type(samples[key]['value']) is torch.Tensor:
                    dist.gather(samples[key]['value'], all_images, dst=0)
                else:
                    dist.gather_object(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    if type(samples[key]['value']) is torch.Tensor:
                        samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples].cpu()
                else:
                    del samples[key]  # Free memory on non-master processes
                torch.cuda.empty_cache()
        
        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            scalars = {}
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )
                elif samples[key]['type'] == 'skeletoned_mesh_list':
                    if samples[key]['value'] is not None:
                        mesh_list = samples[key]['value']
                        import trimesh
                        for mesh in mesh_list:
                            save_dir = os.path.join(self.output_dir, 'samples', suffix, key, mesh['instance'])
                            os.makedirs(save_dir, exist_ok=True)
                            trimesh.Trimesh(vertices=mesh['vertices'].cpu().numpy(), faces=mesh['faces'].cpu().numpy(), process=False).export(save_dir + '/mesh_skl.obj')
                            np.savez(os.path.join(save_dir, 'skeleton.npz'), 
                                     joints=mesh['joints'].cpu().numpy(),
                                     parents=mesh['parents'].cpu().numpy(),
                                     skin=mesh['skin'].cpu().numpy())
                elif samples[key]['type'] == 'joints_parents':
                    if samples[key]['value'] is not None:
                        jp_list = samples[key]['value']
                        for jp in jp_list:
                            # Save N*6 arrays as colored point clouds
                            def save_colored_pcl(array, filename):
                                array = array.detach().cpu().numpy()
                                points = array[:, :3]
                                colors = (array[:, 3:6] * 255).astype(np.uint8)
                                # Combine points and colors into a single array
                                data = np.hstack((points, colors))
                                # Define the .ply header
                                header = f"""ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"""
                                # Write to the .ply file
                                with open(filename, 'w') as file:
                                    file.write(header)
                                    np.savetxt(file, data, fmt='%f %f %f %d %d %d')
                            save_dir = os.path.join(self.output_dir, 'samples', suffix, key, jp['instance'])
                            os.makedirs(save_dir, exist_ok=True)
                            for jp_key in jp.keys():
                                if jp_key != 'instance':
                                    save_colored_pcl(jp[jp_key], os.path.join(save_dir, f'{jp_key}.ply'))
                elif samples[key]['type'] == 'point_cloud':
                    if samples[key]['value'] is not None:
                        pcl_list = samples[key]['value']
                        save_dir = os.path.join(self.output_dir, 'samples', suffix, key)
                        os.makedirs(save_dir, exist_ok=True)
                        for i, item in enumerate(pcl_list):
                            pcl = item['pcl']
                            name = item['instance']  # item.get('instance', f'{i}')
                            filename = os.path.join(save_dir, f'{name}.ply')
                            array = pcl.detach().cpu().numpy()
                            points = array[:, :3]
                            colors = (array[:, 3:6] * 255).astype(np.uint8)
                            data = np.hstack((points, colors))
                            header = f"""ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"""
                            with open(filename, 'w') as file:
                                file.write(header)
                                np.savetxt(file, data, fmt='%f %f %f %d %d %d')
                elif samples[key]['type'] == 'scalar':
                    scalars[key] = samples[key]['value'].mean().item()
                elif samples[key]['type'] == 'text':
                    with open(f'{self.output_dir}/samples/{suffix}/{key}.txt', 'w') as file:
                        for line in samples[key]['value']:
                            file.writelines(line + '\n')
            # If scalars is not empty, save them to a JSON file
            if scalars:
                with open(os.path.join(self.output_dir, 'samples', suffix, 'scalars.json'), 'w') as f:
                    json.dump(scalars, f, indent=4)

        # Delete previous saved samples to save disk space. Keep only the first and the last twos
        samples_dir = os.path.join(self.output_dir, 'samples')
        all_saved = sorted([d for d in os.listdir(samples_dir) if d.startswith('step')])
        if self.is_master and len(all_saved) > 3:
            for saved in all_saved[1:-2]:
                saved_path = os.path.join(samples_dir, saved)
                print(f'Removing previous sample at {saved_path} to save disk space.')
                if os.path.isdir(saved_path):
                    import shutil
                    shutil.rmtree(saved_path)
                else:
                    os.remove(saved_path)

        if self.is_master:
            print(' Done.')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass
    
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0] if hasattr(list(data.values())[0], 'shape') else len(list(data.values())[0])
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        if not self.skip_init_snapshot:
            if self.is_master:
                print('\nStarting training...')
                self.snapshot_dataset()
            if self.step == 0:
                self.snapshot(suffix='init')
            else: # resume
                self.snapshot(suffix=f'resume_step{self.step:07d}')

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            if self.i_sample > 0 and self.step % self.i_sample == 0:
                self.snapshot()

            if self.is_master:
                log.append((self.step, {}))

                # Log time
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if 'amp' in self.fp16_mode:
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all' or self.loss_scaling:
                    log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    ## save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log)}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # show with mlflow
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    if len(log_show) > 0:
                        log_show = dict_reduce(log_show, lambda x: np.mean(x))
                        log_show = dict_flatten(log_show, sep='/')
                        for key, value in log_show.items():
                            self.writer.add_scalar(key, value, self.step)
                    else:
                        print(f'No valid logs to show. Skipping logging at step {self.step}.')
                    log = []

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()

        if self.is_master:
            self.snapshot(suffix='final')
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
            