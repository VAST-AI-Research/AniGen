from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

import utils3d.torch
from ...pipelines import samplers
from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
from ...representations import MeshExtractResult
from ...utils.skin_utils import get_transform
from ...utils.geodesic_noise import maybe_geodesic_smooth_slat_noise
from pytorch3d.ops import knn_points
from ...renderers import MeshRenderer
from ...utils.data_utils import recursive_to_device
import copy


class AniGenSparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
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
        use_joint_num_cond: bool = False,
        p_uncond_joint_num: float = 0.1,
        geodesic_smooth_noise: bool = False,
        geodesic_smooth_noise_iters: int = 0,
        geodesic_smooth_noise_alpha: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_joint_num_cond = use_joint_num_cond
        self.p_uncond_joint_num = p_uncond_joint_num
        self.geodesic_smooth_noise = geodesic_smooth_noise
        self.geodesic_smooth_noise_iters = geodesic_smooth_noise_iters
        self.geodesic_smooth_noise_alpha = geodesic_smooth_noise_alpha
        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 1,
                             "far" : 3}
        self.renderer = MeshRenderer(rendering_options, device=self.device)

    def _maybe_smooth_noise(self, noise):
        return maybe_geodesic_smooth_slat_noise(
            noise,
            self.models['denoiser'],
            enabled=self.geodesic_smooth_noise,
            iters=self.geodesic_smooth_noise_iters,
            alpha=self.geodesic_smooth_noise_alpha,
        )

    def _build_sampler(self, sampler_cls):
        return sampler_cls(
            self.sigma_min,
            geodesic_smooth_noise=self.geodesic_smooth_noise,
            geodesic_smooth_noise_iters=self.geodesic_smooth_noise_iters,
            geodesic_smooth_noise_alpha=self.geodesic_smooth_noise_alpha,
        )
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)
        
    def training_losses(
        self,
        x_0: sp.SparseTensor,
        x_0_skl: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        noise = self._maybe_smooth_noise(noise)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        noise_skl = x_0_skl.replace(torch.randn_like(x_0_skl.feats))
        x_t_skl = self.diffuse(x_0_skl, t, noise=noise_skl)
        cond = self.get_cond(cond, **kwargs)

        local_kwargs = dict(kwargs)
        joints_num = None
        if self.use_joint_num_cond and ('joints_num' in local_kwargs) and (local_kwargs['joints_num'] is not None):
            joints_num = local_kwargs.pop('joints_num')
            if not torch.is_tensor(joints_num):
                joints_num = torch.tensor(joints_num, device=x_0.device)
            joints_num = joints_num.to(device=x_0.device)
            if joints_num.dim() == 0:
                joints_num = joints_num[None].expand(x_0.shape[0])
            joints_num = joints_num.reshape(x_0.shape[0]).clone()

            # IMPORTANT: independent dropout from image/text CFG dropout.
            if self.p_uncond_joint_num and self.p_uncond_joint_num > 0:
                mask_joint = torch.rand(x_0.shape[0], device=x_0.device) < float(self.p_uncond_joint_num)
                joints_num[mask_joint] = 0

        if self.use_joint_num_cond:
            pred, pred_skl = self.training_models['denoiser'](x_t, x_t_skl, t * 1000, cond, joints_num=joints_num, **local_kwargs)
        else:
            pred, pred_skl = self.training_models['denoiser'](x_t, x_t_skl, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        assert pred_skl.shape == noise_skl.shape == x_0_skl.shape
        target = self.get_v(x_0, noise, t)
        target_skl = self.get_v(x_0_skl, noise_skl, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["mse_skl"] = F.mse_loss(pred_skl.feats, target_skl.feats)
        terms["loss"] = terms["mse"] + terms["mse_skl"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        mse_skl_per_instance = np.array([
            F.mse_loss(pred_skl.feats[x_0_skl.layout[i]], target_skl.feats[x_0_skl.layout[i]]).item()
            for i in range(x_0_skl.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
                terms[f"bin_{i}_skl"] = {"mse": mse_skl_per_instance[time_bin == i].mean()}

        return terms, {}

    def get_sampler(self, **kwargs) -> samplers.AniGenFlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return self._build_sampler(samplers.AniGenFlowEulerSampler)
    
    @torch.no_grad()
    def _flip_normal(self, normal: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Flip normal to align with camera.
        """
        normal = normal * 2.0 - 1.0
        R = torch.zeros_like(extrinsics)
        R[:, :3, :3] = extrinsics[:, :3, :3]
        R[:, 3, 3] = 1.0
        view_dir = utils3d.torch.unproject_cv(
            utils3d.torch.image_uv(*normal.shape[-2:], device=self.device).reshape(1, -1, 2),
            torch.ones(*normal.shape[-2:], device=self.device).reshape(1, -1),
            R, intrinsics
        ).reshape(-1, *normal.shape[-2:], 3).permute(0, 3, 1, 2)
        unflip = (normal * view_dir).sum(1, keepdim=True) < 0
        normal *= unflip * 2.0 - 1.0
        return (normal + 1.0) / 2.0
    
    def _render_batch(self, reps: List[MeshExtractResult], extrinsics: torch.Tensor, intrinsics: torch.Tensor, return_types=['mask', 'normal', 'depth'], specified_colors=None) -> Dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_types: vary in ['mask', 'normal', 'depth', 'normal_map', 'color']
            
        Returns: 
            a dict with
                reg_loss : [N] tensor of regularization losses
                mask : [N x 1 x H x W] tensor of rendered masks
                normal : [N x 3 x H x W] tensor of rendered normals
                depth : [N x 1 x H x W] tensor of rendered depths
        """
        if extrinsics.shape[0] == 1:
            extrinsics = extrinsics.expand(len(reps), -1, -1)
        if intrinsics.shape[0] == 1:
            intrinsics = intrinsics.expand(len(reps), -1, -1)
        ret = {k : [] for k in return_types}
        for i, rep in enumerate(reps):
            specified_color = None if specified_colors is None else specified_colors[i]
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types, specified_color=specified_color)
            for k in out_dict:
                ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret

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
        ret_dict = {}
        cond_vis = []
        gt_meshes, gt_reps, gt_reps_skl, reps, reps_skl, instances = [], [], [], [], [], []
        joints_gt, parents_gt, skins_gt = [], [], []
        gt_skin_colors, pred_skin_colors, gt_recon_skin_colors, pred_skin_colors_with_gt_skl, pred_skin_colors_with_gt_ss = [], [], [], [], []
        incorrect_grouping_instances = []
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            for key in data:
                if type(data[key]) is list and hasattr(data[key][0], 'device'):
                    for j in range(len(data[key])):
                        data[key][j] = data[key][j].to(self.device)
            data = recursive_to_device(data, self.device)
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            noise = self._maybe_smooth_noise(noise)
            noise_skl = data['x_0_skl'].replace(torch.randn_like(data['x_0_skl'].feats))
            gt_rep, gt_rep_skl = self.dataset.decode_latent(data['x_0'], data['x_0_skl'])
            gt_reps.extend(gt_rep)
            gt_reps_skl.extend(gt_rep_skl)
            gt_meshes.extend(data['mesh'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            del data['x_0_skl']
            args = self.get_inference_cond(**data)

            # `get_inference_cond(**data)` already forwards extra kwargs, including `joints_num`.
            if self.use_joint_num_cond and ('joints_num' in args) and (args['joints_num'] is not None):
                # If CFG is active (neg_cond provided), make unconditional joints_num explicit.
                if 'neg_cond' in args and 'neg_joints_num' not in args:
                    jn = args['joints_num']
                    args['neg_joints_num'] = torch.zeros_like(jn) if torch.is_tensor(jn) else 0

            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                noise_skl=noise_skl,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            rep, rep_skl, skins_gt_sskin, skins_gt_sklskin = self.dataset.decode_latent(res.samples.cuda(), res.samples_skl.cuda(), gt_reps=gt_rep, gt_reps_skl=gt_rep_skl)
            reps.extend(rep)
            reps_skl.extend(rep_skl)
            instances.extend(data['instance'])
            joints_gt.extend(data['joints'])
            parents_gt.extend(data['parents'])
            skins_gt.extend(data['skin'])

            jmapping_gtrec2gt = [knn_points(gt_j[None], gt_rep_skl_.joints_grouped[None], K=1)[1][0, :, 0] for gt_rep_skl_, gt_j in zip(gt_rep_skl, data['joints'])]
            jmapping_gen2gt   = [knn_points(gt_j[None], rep_skl_.joints_grouped[None], K=1)[1][0, :, 0]    for rep_skl_, gt_j in zip(rep_skl, data['joints'])]

            # Skin color for GT mesh and pred mesh
            skin_color_trans_funcs = [get_transform(skin) for skin in data['skin']]
            gt_skin_colors.extend(              [func(skin) for func, skin in zip(skin_color_trans_funcs, data['skin'])])
            pred_skin_colors_with_gt_ss.extend( [func(skin[:, jmap]) for func, skin, jmap in zip(skin_color_trans_funcs, skins_gt_sskin,                         jmapping_gen2gt)])
            pred_skin_colors_with_gt_skl.extend([func(skin[:, jmap]) for func, skin, jmap in zip(skin_color_trans_funcs, skins_gt_sklskin,                       jmapping_gtrec2gt)])
            gt_recon_skin_colors.extend(        [func(skin[:, jmap]) for func, skin, jmap in zip(skin_color_trans_funcs, [rs['skin_pred'] for rs in gt_rep_skl], jmapping_gtrec2gt)])
            pred_skin_colors.extend(            [func(skin[:, jmap]) for func, skin, jmap in zip(skin_color_trans_funcs, [rs['skin_pred'] for rs in rep_skl],    jmapping_gen2gt)])
            
            jmapping_gen2gtrec = [knn_points(gt_rep_skl_.joints_grouped[None], rep_skl_.joints_grouped[None], K=1)[1][0, :, 0] for rep_skl_, gt_rep_skl_ in zip(rep_skl, gt_rep_skl)]
            incorrect_grouping_instances.extend([ins for ins, jmap, rep_skl_ in zip(args['instance'], jmapping_gen2gtrec, rep_skl) if not (torch.bincount(jmap, minlength=rep_skl_.joints_grouped.shape[0]) == 1).all()])

        ret_dict.update({f'incorrect_grouping_instances': {'value': incorrect_grouping_instances, 'type': 'text'}})

        # Build camera
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = 512
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]
        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)
        exts = torch.stack(exts, dim=0)[0]
        ints = torch.stack(ints, dim=0)[0]

        # GT
        return_types = ['normal', 'specified']
        gt_render_results = self._render_batch([
            MeshExtractResult(vertices=mesh['vertices'].to(self.device), faces=mesh['faces'].to(self.device))
            for mesh in gt_meshes
        ], exts[None], ints[None], return_types=return_types, specified_colors=gt_skin_colors)
        ret_dict.update({f'gt_normal': {'value': self._flip_normal(gt_render_results['normal'], exts[None], ints[None]), 'type': 'image'}})
        ret_dict.update({f'gt_skin_map': {'value': gt_render_results['specified'], 'type': 'image'}})
        # GT Recon
        return_types = ['normal', 'color', 'normal_map', 'specified']
        render_results = self._render_batch(gt_reps, exts[None], ints[None], return_types=return_types, specified_colors=gt_recon_skin_colors)
        ret_dict.update({f'gt_rec_normal': {'value': render_results['normal'], 'type': 'image'}})
        ret_dict.update({f'gt_rec_image': {'value': render_results['color'], 'type': 'image'}})
        ret_dict.update({f'gt_rec_normal_map': {'value': render_results['normal_map'], 'type': 'image'}})
        ret_dict.update({f'gt_rec_skin_map': {'value': render_results['specified'], 'type': 'image'}})
        # Pred
        return_types = ['normal', 'color', 'normal_map', 'specified']
        render_results = self._render_batch(reps, exts[None], ints[None], return_types=return_types, specified_colors=pred_skin_colors)
        ret_dict.update({f'gen_normal': {'value': render_results['normal'], 'type': 'image'}})
        ret_dict.update({f'gen_image': {'value': render_results['color'], 'type': 'image'}})
        ret_dict.update({f'gen_normal_map': {'value': render_results['normal_map'], 'type': 'image'}})
        ret_dict.update({f'gen_skin_map': {'value': render_results['specified'], 'type': 'image'}})
        # Pred with GT skl skin
        render_results = self._render_batch(reps, exts[None], ints[None], return_types=['specified'], specified_colors=pred_skin_colors_with_gt_skl)
        ret_dict.update({f'gen_skin_map_with_gt_skl': {'value': render_results['specified'], 'type': 'image'}})
        # Pred with GT ss skin
        render_results = self._render_batch(gt_reps, exts[None], ints[None], return_types=['specified'], specified_colors=pred_skin_colors_with_gt_ss)
        ret_dict.update({f'gen_skin_map_with_gt_ss': {'value': render_results['specified'], 'type': 'image'}})
    
        ret_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        ## Skeletoned Meshes
        skeletoned_meshes = []
        joints_parents = []
        for i, rep_skl in enumerate(reps_skl):
            ##################################
            # Predicted mesh with GT skeleton
            ##################################
            skeletoned_mesh = dict()
            skeletoned_mesh['instance'] = instances[i]
            vertices_pred, faces_pred = reps[i].vertices, reps[i].faces
            skeletoned_mesh['vertices'] = vertices_pred
            skeletoned_mesh['faces'] = faces_pred
            skeletoned_mesh['joints'] = rep_skl.joints_grouped
            skeletoned_mesh['parents'] = rep_skl.parents_grouped
            skeletoned_mesh['skin'] = rep_skl.skin_pred
            skeletoned_meshes.append(skeletoned_mesh)

            ##################################
            # Predicted mesh with GT skeleton
            ##################################
            joint_parent = dict()
            joint_parent['instance'] = instances[i]
            # GT joints
            joint_gt, parent_idx_gt = joints_gt[i], parents_gt[i]
            import matplotlib.pyplot as plt
            joint_colors_gt = torch.tensor(plt.cm.get_cmap('tab20')(np.arange(len(joint_gt)))[:, :3], dtype=joint_gt.dtype, device=joint_gt.device)
            joint_gt_with_color = torch.cat([joint_gt, joint_colors_gt], dim=-1)
            # GT parents
            parent_gt = joint_gt[parent_idx_gt[1:]]
            parent_colors_gt = joint_colors_gt[parent_idx_gt[1:]]
            parent_gt_with_color = torch.cat([parent_gt, parent_colors_gt], dim=-1)
            # Predicted joints
            position_skl = rep_skl.positions
            _, joint_nn_idx, _ = knn_points(position_skl[None], joint_gt[None], K=1, norm=2, return_nn=False)
            joint_nn_idx = joint_nn_idx[0, :, 0]
            joint_pred = rep_skl.joints
            joint_colors_pred = joint_colors_gt[joint_nn_idx]
            joint_pred_with_color = torch.cat([joint_pred, joint_colors_pred], dim=-1)
            # Predicted parents
            parent_pred = rep_skl.parents
            parents_nn_idx = parent_idx_gt[joint_nn_idx]
            parents_nn_idx[parents_nn_idx == -1] = 0
            parent_colors_pred = joint_colors_gt[parents_nn_idx]
            parent_pred_with_color = torch.cat([parent_pred, parent_colors_pred], dim=-1)
            # Combine
            joint_parent['joints_gt'] = joint_gt_with_color
            joint_parent['parents_gt'] = parent_gt_with_color
            joint_parent['joints_pred'] = joint_pred_with_color
            joint_parent['parents_pred'] = parent_pred_with_color
            # Confidence
            if rep_skl.conf_j is not None:
                color_conf_j = rep_skl.conf_j.expand(-1, 1)
                color_conf_j = (color_conf_j - color_conf_j.min()) / (color_conf_j.max() - color_conf_j.min() + 1e-8)
                color_conf_j = torch.cat([color_conf_j, torch.zeros_like(color_conf_j), 1-color_conf_j], dim=-1)
                joint_parent['joints_conf'] = torch.cat([joint_pred, color_conf_j], dim=-1)
            if rep_skl.conf_p is not None:
                color_conf_p = rep_skl.conf_p.expand(-1, 1)
                color_conf_p = (color_conf_p - color_conf_p.min()) / (color_conf_p.max() - color_conf_p.min() + 1e-8)
                color_conf_p = torch.cat([color_conf_p, torch.zeros_like(color_conf_p), 1-color_conf_p], dim=-1)
                joint_parent['parents_conf'] = torch.cat([parent_pred, color_conf_p], dim=-1)
            if rep_skl.joints_grouped is not None:
                joint_parent['joints_grouped'] = torch.cat([rep_skl.joints_grouped, torch.ones_like(rep_skl.joints_grouped)], dim=-1)
            if rep_skl.parents_grouped is not None:
                parents_grouped = rep_skl.joints_grouped[rep_skl.parents_grouped]
                joint_parent['parents_grouped'] = torch.cat([parents_grouped, torch.ones_like(parents_grouped)], dim=-1)
            joints_parents.append(joint_parent)
    
        ret_dict.update({
            'skeletoned_meshes': {'value': skeletoned_meshes, 'type': 'skeletoned_mesh_list'},
            'joints_parents':    {'value': joints_parents,    'type': 'joints_parents'},
        })
        
        return ret_dict


class AniGenSparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, AniGenSparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
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
    
    def get_sampler(self, **kwargs) -> samplers.AniGenFlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process with classifier-free guidance.
        """
        return self._build_sampler(samplers.AniGenFlowEulerCfgSampler)


class AniGenTextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, AniGenSparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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
    pass


class AniGenImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, AniGenSparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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
    pass
