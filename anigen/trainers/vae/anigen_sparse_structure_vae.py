from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer


class AniGenSparseStructureVaeTrainer(BasicTrainer):
    """
    Trainer for Sparse Structure VAE.
    
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
        
        loss_type (str): Loss type. 'bce' for binary cross entropy, 'l1' for L1 loss, 'dice' for Dice loss.
        lambda_kl (float): KL divergence loss weight.
    """
    
    def __init__(
        self,
        *args,
        loss_type='bce',
        lambda_kl=1e-3,
        lambda_kl_skl=1e-3,
        latent_denoising=False,
        latent_denoising_gamma=1.0,
        latent_denoising_skl=False,
        latent_denoising_gamma_skl=1.0,
        latent_time_max=0.5,
        latent_time_max_skl=0.5,
        latent_achive_max_step=0,
        latent_achive_max_step_skl=0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.lambda_kl_skl = lambda_kl_skl
        self.latent_denoising = latent_denoising
        self.latent_denoising_gamma = latent_denoising_gamma
        self.latent_denoising_skl = latent_denoising_skl
        self.latent_denoising_gamma_skl = latent_denoising_gamma_skl
        self.latent_time_max = latent_time_max
        self.latent_time_max_skl = latent_time_max_skl
        self.latent_achive_max_step = latent_achive_max_step
        self.latent_achive_max_step_skl = latent_achive_max_step_skl
    
    def training_losses(
        self,
        ss: torch.Tensor,
        ss_skl: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            ss: The [N x 1 x H x W x D] tensor of binary sparse structure.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """

        z, mean, logvar, z_skl, mean_skl, logvar_skl = self.training_models['encoder'](ss.float(), ss_skl.float(), sample_posterior=True, return_raw=True)

        terms = edict(loss = 0.0)
        if self.latent_denoising:
            # Progressively increase the maximum time
            latent_time_max = min(1, (self.step / self.latent_achive_max_step)) * self.latent_time_max if self.latent_achive_max_step > 0 else self.latent_time_max
            noise = torch.randn_like(z) * self.latent_denoising_gamma
            time = torch.rand(z.shape[0], *[1] * (len(z.shape) -1)).to(z)
            time = (1 - (1 - time).clip(min=1e-8).sqrt()) * latent_time_max
            z = (1 - time) * z + time * noise
        else:
            terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
            terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        if self.latent_denoising_skl:
            # Progressively increase the maximum time
            latent_time_max_skl = min(1, (self.step / self.latent_achive_max_step_skl)) * self.latent_time_max_skl if self.latent_achive_max_step_skl > 0 else self.latent_time_max_skl
            noise_skl = torch.randn_like(z_skl) * self.latent_denoising_gamma_skl
            time_skl = torch.rand(z_skl.shape[0], *[1] * (len(z_skl.shape) -1)).to(z_skl)
            time_skl = (1 - (1 - time_skl).clip(min=1e-8).sqrt()) * latent_time_max_skl
            z_skl = (1 - time_skl) * z_skl + time_skl * noise_skl
        else:
            if mean_skl is not None and logvar_skl is not None:
                terms["kl_skl"] = 0.5 * torch.mean(mean_skl.pow(2) + logvar_skl.exp() - logvar_skl - 1)
                terms["loss"] = terms["loss"] + self.lambda_kl_skl * terms["kl_skl"]

        logits, logits_skl = self.training_models['decoder'](z, z_skl)

        if self.loss_type == 'bce':
            terms["bce"] = F.binary_cross_entropy_with_logits(logits, ss.float(), reduction='mean')
            terms["bce_skl"] = F.binary_cross_entropy_with_logits(logits_skl, ss_skl.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["bce"] + terms["bce_skl"]
        elif self.loss_type == 'l1':
            terms["l1"] = F.l1_loss(F.sigmoid(logits), ss.float(), reduction='mean')
            terms["l1_skl"] = F.l1_loss(F.sigmoid(logits_skl), ss_skl.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["l1"] + terms["l1_skl"]
        elif self.loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms["dice"] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
            logits_skl = F.sigmoid(logits_skl)
            terms["dice_skl"] = 1 - (2 * (logits_skl * ss_skl.float()).sum() + 1) / (logits_skl.sum() + ss_skl.float().sum() + 1)
            terms["loss"] = terms["loss"] + terms["dice"] + terms["dice_skl"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        return terms, {}
    
    @torch.no_grad()
    def snapshot(self, *args, **kwargs):
        return super().snapshot(*args, **kwargs)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        disturbance: float = 0.0,
        **kwargs
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        gts = []
        recons = []
        gts_skl = []
        recons_skl = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            z, z_skl = self.models['encoder'](args['ss'].float(), args['ss_skl'].float(), sample_posterior=False)
            if disturbance > 0.0:
                noise = torch.randn_like(z)
                t = torch.rand([z.shape[0], *[1] * (len(z.shape) -1)]).to(z) * disturbance
                z = (1 - t) * z + t * noise
                noise_skl = torch.randn_like(z_skl)
                t_skl = torch.rand([z_skl.shape[0], *[1] * (len(z_skl.shape) -1)]).to(z_skl) * disturbance
                z_skl = (1 - t_skl) * z_skl + t_skl * noise_skl
            logits, logits_skl = self.models['decoder'](z, z_skl)

            recon = (logits > 0).long()
            gts.append(args['ss'])
            recons.append(recon)

            recon_skl = (logits_skl > 0).long()
            gts_skl.append(args['ss_skl'])
            recons_skl.append(recon_skl)

        sample_dict = {
            'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'},
            'recon': {'value': torch.cat(recons, dim=0), 'type': 'sample'},
            'gt_skl': {'value': torch.cat(gts_skl, dim=0), 'type': 'sample'},
            'recon_skl': {'value': torch.cat(recons_skl, dim=0), 'type': 'sample'},
        }
        return sample_dict
