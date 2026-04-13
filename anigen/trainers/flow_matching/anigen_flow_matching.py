from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class AniGenFlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
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

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,

        # inpaint training (x0 known) controls
        train_inpaint: bool = False,
        p_x0_inject: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

        self.train_inpaint = bool(train_inpaint)
        self.p_x0_inject = float(p_x0_inject)

    def diffuse(self, x_0: torch.Tensor, x_0_skl: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, noise_skl: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        if noise_skl is None:
            noise_skl = torch.randn_like(x_0_skl)

        assert noise.shape == x_0.shape, "noise must have same shape as x_0"
        assert noise_skl.shape == x_0_skl.shape, "noise_skl must have same shape as x_0_skl"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        x_t_skl = (1 - t) * x_0_skl + (self.sigma_min + (1 - self.sigma_min) * t) * noise_skl

        return x_t, x_t_skl

    def reverse_diffuse(self, x_t: torch.Tensor, x_t_skl: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, noise_skl: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        assert noise_skl.shape == x_t_skl.shape, "noise_skl must have same shape as x_t_skl"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        x_0_skl = (x_t_skl - (self.sigma_min + (1 - self.sigma_min) * t) * noise_skl) / (1 - t)
        return x_0, x_0_skl

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.AniGenFlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.AniGenFlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        x_0_skl: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        noise_skl = torch.randn_like(x_0_skl)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t, x_t_skl = self.diffuse(x_0=x_0, x_0_skl=x_0_skl, t=t, noise=noise, noise_skl=noise_skl)
        cond = self.get_cond(cond, **kwargs)

        if self.train_inpaint:
            kwargs = dict(kwargs)
            kwargs.update({
                'x0': x_0,
            })
        
        pred, pred_skl = self.training_models['denoiser'](x_t, x_t_skl, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        assert pred_skl.shape == noise_skl.shape == x_0_skl.shape
        
        target = self.get_v(x_0, noise, t)
        target_skl = self.get_v(x_0_skl, noise_skl, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]
        terms["mse_skl"] = F.mse_loss(pred_skl, target_skl)
        terms["loss"] = terms["loss"] + terms["mse_skl"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        mse_skl_per_instance = np.array([
            F.mse_loss(pred_skl[i], target_skl[i]).item()
            for i in range(x_0_skl.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
                terms[f"bin_{i}"]["mse_skl"] = mse_skl_per_instance[time_bin == i].mean()

        return terms, {}
    
    def voxel2pcl(self, voxels, voxels_skl=None, names=None, density=3, threshold=0.5):
        if hasattr(self.dataset, 'decode_latent') and voxels_skl is not None:
            voxels, voxels_skl = self.dataset.decode_latent(voxels, voxels_skl)
        def _convert(v):
            if v.dim() == 5:
                v = v[:, 0]
            N, D, H, W = v.shape
            device = v.device
            offset = (torch.arange(density, device=device) + 0.5) / density
            oz, oy, ox = torch.meshgrid(offset, offset, offset, indexing='ij')
            offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3) # (density^3, 3)
            pcls = []
            for i in range(N):
                occupied = (v[i] > threshold).nonzero() # (M, 3) -> (z, y, x)
                if occupied.shape[0] == 0:
                    pcl = torch.zeros((0, 6), device=device)
                else:
                    z_idx, y_idx, x_idx = occupied[:, 0], occupied[:, 1], occupied[:, 2]
                    base_coords = torch.stack([x_idx, y_idx, z_idx], dim=-1).float()
                    points = base_coords.unsqueeze(1) + offsets.unsqueeze(0)
                    points = points.reshape(-1, 3)
                    scale = torch.tensor([W, H, D], device=device).float()
                    points = points / scale - 0.5
                    colors = points + 0.5
                    pcl = torch.cat([points, colors], dim=-1)
                if names is not None:
                    pcls.append({'instance': names[i], 'pcl': pcl})
                else:
                    pcls.append(pcl)
            return pcls
        pcls = _convert(voxels)
        if voxels_skl is not None:
            pcls_skl = _convert(voxels_skl)
            return pcls, pcls_skl
        return pcls

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        **kwargs,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample_gt_skl = []
        sample = []
        sample_skl = []
        cond_vis = []
        instances = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            instances.extend(data['instance'][:batch])

            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            noise_skl = torch.randn_like(data['x_0_skl'])
            sample_gt.append(data['x_0'])
            sample_gt_skl.append(data['x_0_skl'])
            cond_vis.append(self.vis_cond(**data))
            data['x0'] = data['x_0']
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                noise_skl=noise_skl,
                **args,
                steps=50, cfg_strength=0 if self.train_inpaint else 3.0, verbose=verbose,
            )
            sample.append(res.samples)
            sample_skl.append(res.samples_skl)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_gt_skl = torch.cat(sample_gt_skl, dim=0)
        sample_skl = torch.cat(sample_skl, dim=0)

        sample_pcl, sample_skl_pcl = self.voxel2pcl(sample, sample_skl, names=instances)

        sample_dict = {
            'sample_gt': {'value': {'x_0': sample_gt, 'x_0_skl': sample_gt_skl}, 'type': 'sample'},
            'sample': {'value': {'x_0': sample, 'x_0_skl': sample_skl}, 'type': 'sample'},
            'sample_pcl': {'value': sample_pcl, 'type': 'point_cloud'},
            'sample_skl_pcl': {'value': sample_skl_pcl, 'type': 'point_cloud'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    
class AniGenFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, AniGenFlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
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

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """

    def get_sampler(self, **kwargs) -> samplers.AniGenFlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.AniGenFlowEulerSampler(self.sigma_min)


class AniGenTextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, AniGenFlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """

    def get_sampler(self, **kwargs) -> samplers.AniGenFlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.AniGenFlowEulerSampler(self.sigma_min)


class AniGenImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, AniGenFlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """

    def get_sampler(self, **kwargs) -> samplers.AniGenFlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.AniGenFlowEulerSampler(self.sigma_min)
