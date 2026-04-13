from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch
from ...renderers import MeshRenderer
from ...representations import MeshExtractResult
from ...utils.data_utils import recursive_to_device
from ...utils.skin_utils import get_transform

from ..basic import BasicTrainer

from ...representations import Gaussian
from ...renderers import GaussianRenderer
from ...representations.octree import DfsOctree as Octree
from ...renderers import OctreeRenderer

from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, smooth_l1_loss, l2_loss, ssim, lpips
from pytorch3d.ops import knn_points
import torch.nn.functional as F


class AniGenSLatVaeSkeletonTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE.
    
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
        
        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        regularizations (dict): Regularization config.
    """
    
    def __init__(
        self,
        *args,
        depth_loss_type: str = 'l1',
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_depth: int = 1,
        lambda_tsdf: float = 0.01,
        lambda_color: float = 0.0,
        lambda_kl: float = 1e-3,
        lambda_kl_skl: float = 1e-3,

        alpha_conf_j: float = 0.1,
        alpha_conf_p: float = 0.1,

        lambda_joints: float = 1.0,
        lambda_parents: float = 1.0,
        lambda_skin_kl: float = 0.5,
        lambda_skin_feats_l2: float = 1.0,
        lambda_skin_feats_l2_skl: float = 1.0,
        lambda_skin_var: float = 1.0,

        latent_denoising=False,
        latent_denoising_gamma=1.0,
        latent_denoising_skl=False,
        latent_denoising_gamma_skl=1.0,
        latent_time_max=0.5,
        latent_time_max_skl=0.5,
        latent_achive_max_step=0,
        latent_achive_max_step_skl=0,

        regularizations: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.depth_loss_type = depth_loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_depth = lambda_depth
        self.lambda_tsdf = lambda_tsdf
        self.lambda_color = lambda_color
        self.lambda_kl = lambda_kl
        self.lambda_kl_skl = lambda_kl_skl
        self.alpha_conf_j = alpha_conf_j
        self.alpha_conf_p = alpha_conf_p
        self.lambda_joints = lambda_joints
        self.lambda_parents = lambda_parents
        self.lambda_skin_kl = lambda_skin_kl
        self.lambda_skin_feats_l2 = lambda_skin_feats_l2
        self.lambda_skin_feats_l2_skl = lambda_skin_feats_l2_skl
        self.lambda_skin_var = lambda_skin_var

        self.latent_denoising = latent_denoising
        self.latent_denoising_gamma = latent_denoising_gamma
        self.latent_denoising_skl = latent_denoising_skl
        self.latent_denoising_gamma_skl = latent_denoising_gamma_skl
        self.latent_time_max = latent_time_max
        self.latent_time_max_skl = latent_time_max_skl
        self.latent_achive_max_step = latent_achive_max_step
        self.latent_achive_max_step_skl = latent_achive_max_step_skl

        self.regularizations = regularizations
        self.use_color = self.lambda_color > 0
        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 1,
                             "far" : 3}
        self.renderer = MeshRenderer(rendering_options, device=self.device)

    @torch.no_grad()
    def render_coords(self, coords_list, extrinsics: torch.Tensor, intrinsics: torch.Tensor, colors_overwrite_list=None, ss_resolution=64):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        images = []
        for i in range(len(coords_list)):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = coords_list[i][:, 1:]
            representation.position = coords.float() / ss_resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(ss_resolution)), dtype=torch.uint8, device='cuda')

            image = renderer.render(representation, extrinsics[i], intrinsics[i], colors_overwrite=representation.position if colors_overwrite_list is None else colors_overwrite_list[i])['color']
            images.append(image)
        return torch.stack(images, dim=0)
    
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
        ret = {k : [] for k in return_types}
        for i, rep in enumerate(reps):
            specified_color = None if specified_colors is None else specified_colors[i]
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types, specified_color=specified_color)
            for k in out_dict:
                ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret
    
    @staticmethod
    def _tsdf_reg_loss(rep: MeshExtractResult, depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        # Calculate tsdf
        with torch.no_grad():
            # Project points to camera and calculate pseudo-sdf as difference between gt depth and projected depth
            projected_pts, pts_depth = utils3d.torch.project_cv(extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v)
            projected_pts = (projected_pts - 0.5) * 2.0
            depth_map_res = depth_map.shape[1]
            gt_depth = torch.nn.functional.grid_sample(depth_map.reshape(1, 1, depth_map_res, depth_map_res), projected_pts.reshape(1, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=True)
            pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
            # Truncate pseudo-sdf
            delta = 1 / rep.res * 3.0
            trunc_mask = pseudo_sdf > -delta
        
        # Loss
        gt_tsdf = pseudo_sdf[trunc_mask]
        tsdf = rep.tsdf_s.flatten()[trunc_mask]
        gt_tsdf = torch.clamp(gt_tsdf, -delta, delta)
        return torch.mean((tsdf - gt_tsdf) ** 2)

    def _calc_tsdf_loss(self, reps : list[MeshExtractResult], depth_maps, extrinsics, intrinsics) -> torch.Tensor:
        tsdf_loss = 0.0
        for i, rep in enumerate(reps):
            tsdf_loss += self._tsdf_reg_loss(rep, depth_maps[i], extrinsics[i], intrinsics[i])
        return tsdf_loss / len(reps)
    
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

    def _perceptual_loss(self, gt: torch.Tensor, pred: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
        """
        Combination of L1, SSIM, and LPIPS loss.
        """
        if gt.shape[1] != 3:
            assert gt.shape[-1] == 3
            gt = gt.permute(0, 3, 1, 2)
        if pred.shape[1] != 3:
            assert pred.shape[-1] == 3
            pred = pred.permute(0, 3, 1, 2)
        terms = {
            f"{name}_loss" : l1_loss(gt, pred),
            f"{name}_loss_ssim" : 1 - ssim(gt, pred),
            f"{name}_loss_lpips" : lpips(gt, pred)
        }
        terms[f"{name}_loss_perceptual"] = terms[f"{name}_loss"] + terms[f"{name}_loss_ssim"] * self.lambda_ssim + terms[f"{name}_loss_lpips"] * self.lambda_lpips
        return terms
    
    def geometry_losses(
        self,
        reps: List[MeshExtractResult],
        mesh: List[Dict],
        normal_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        with torch.no_grad():
            gt_meshes = []
            for i in range(len(reps)):
                gt_mesh = MeshExtractResult(mesh[i]['vertices'].to(self.device), mesh[i]['faces'].to(self.device))
                gt_meshes.append(gt_mesh)
            target = self._render_batch(gt_meshes, extrinsics, intrinsics, return_types=['mask', 'depth', 'normal'])
            target['normal'] = self._flip_normal(target['normal'], extrinsics, intrinsics)
                
        terms = edict(geo_loss = 0.0)
        if self.lambda_tsdf > 0:
            tsdf_loss = self._calc_tsdf_loss(reps, target['depth'], extrinsics, intrinsics)
            terms['tsdf_loss'] = tsdf_loss
            terms['geo_loss'] += tsdf_loss * self.lambda_tsdf
        
        return_types = ['mask', 'depth', 'normal', 'normal_map'] if self.use_color else ['mask', 'depth', 'normal']
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
        
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            for k, v in buffer.items():
                buffer[k] = v[success_mask]
            for k, v in target.items():
                target[k] = v[success_mask]
            
            terms['mask_loss'] = l1_loss(buffer['mask'], target['mask']) 
            if self.depth_loss_type == 'l1':
                terms['depth_loss'] = l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'])
            elif self.depth_loss_type == 'smooth_l1':
                terms['depth_loss'] = smooth_l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'], beta=1.0 / (2 * reps[0].res))
            else:
                raise ValueError(f"Unsupported depth loss type: {self.depth_loss_type}")
            terms.update(self._perceptual_loss(buffer['normal'] * target['mask'], target['normal'] * target['mask'], 'normal'))
            terms['geo_loss'] = terms['geo_loss'] + terms['mask_loss'] + terms['depth_loss'] * self.lambda_depth + terms['normal_loss_perceptual']
            if self.use_color and normal_map is not None:
                terms.update(self._perceptual_loss(normal_map[success_mask], buffer['normal_map'], 'normal_map'))
                terms['geo_loss'] = terms['geo_loss'] + terms['normal_map_loss_perceptual'] * self.lambda_color
                
        return terms
      
    def color_losses(self, reps, image, alpha, extrinsics, intrinsics):
        terms = edict(color_loss = torch.tensor(0.0, device=self.device))
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=['color'])
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            terms.update(self._perceptual_loss(image * alpha[:, None][success_mask], buffer['color'][success_mask], 'color'))
            terms['color_loss'] = terms['color_loss'] + terms['color_loss_perceptual'] * self.lambda_color
        return terms
    
    def skeleton_losses(self, joints_list, parents_list, skin_list, is_bad_skin_list, reps, reps_skl, gt_meshes, joint_skin_embeds_gt_list, vert_skin_embeds_gt_list, cache_gt_skin_embeds=False, cache_gt_nn_skin=False, **kwargs):
        terms = edict(
            joints_loss = torch.tensor(0.0, device=self.device),
            parents_loss = torch.tensor(0.0, device=self.device),
            skin_feats_joints_var_loss =torch.tensor(0.0, device=self.device),
            skin_kl_loss = torch.tensor(0.0, device=self.device),
            skin_feats_l2_loss_vert = torch.tensor(0.0, device=self.device),
            skin_feats_l2_loss_joints = torch.tensor(0.0, device=self.device),
        )
        for i, rep, rep_skl in zip(range(len(reps)), reps, reps_skl):
            joints_gt, parents_gt, skin_gt, is_bad_skin = joints_list[i], parents_list[i], skin_list[i], is_bad_skin_list[i]
            gt_mesh = gt_meshes[i]

            # Read predicted skeleton data
            joints_pred = rep_skl.joints
            parents_pred = rep_skl.parents
            positions_skl = rep_skl.positions

            ##########################
            # Joint and Parent Loss
            ##########################
            # Calculate GT joints and parents
            dist_nn_12, joints_nn_idx, _ = knn_points(positions_skl[None], joints_gt[None], K=2, norm=2, return_nn=False)
            joints_nn_idx = joints_nn_idx[0, :, 0]
            joints_nn_gt = joints_gt[joints_nn_idx]
            parents_nn_idx = parents_gt[joints_nn_idx]
            parents_nn_idx[parents_nn_idx == -1] = 0
            parents_nn_gt = joints_gt[parents_nn_idx]

            # Calculate NN dist between joints to weight the parents loss
            nn_dist_joints, _, _ = knn_points(joints_gt[None], joints_gt[None], K=2, norm=2, return_nn=False)
            nn_dist_joints = nn_dist_joints[0, :, 1]
            dist_weights = (nn_dist_joints.max().clamp(min=1e-6) / nn_dist_joints.clamp(min=1e-6)).clamp(min=1.0, max=10.0)
            joints_weights = dist_weights[joints_nn_idx]
            parents_weights = dist_weights[parents_nn_idx]

            # Confidence loss of joint and parent predictions
            if rep_skl.conf_j is None or rep_skl.conf_p is None:
                joints_loss = (joints_pred - joints_nn_gt).abs() * joints_weights[:, None]
                parents_loss = (parents_pred - parents_nn_gt).abs() * parents_weights[:, None]
            else:
                if rep_skl.jp_hyper_continuous:
                    factor = (1 - (dist_nn_12[0, :, 0:1] / (dist_nn_12[0, :, 1:2] + 1e-8)).clamp(max=1.0)).clamp(min=0.1)
                    conf_j = conf_p = factor
                    joint_conf_loss  = (rep_skl.conf_j - factor).abs().mean()
                    parent_conf_loss = (rep_skl.conf_p - factor).abs().mean()
                else:
                    conf_j =  torch.exp(rep_skl.conf_j) + 1
                    conf_p = torch.exp(rep_skl.conf_p) + 1
                    joint_conf_loss = - torch.maximum((2**5) * torch.ones_like(conf_j), torch.log(conf_j)) * self.alpha_conf_j
                    parent_conf_loss =  - torch.maximum((2**5) * torch.ones_like(conf_p), torch.log(conf_p)) * self.alpha_conf_p
                    joint_conf_loss = joint_conf_loss - joint_conf_loss.detach()
                    parent_conf_loss = parent_conf_loss - parent_conf_loss.detach()
                joints_loss  = conf_j * (joints_pred - joints_nn_gt).abs() * joints_weights[:, None]  + joint_conf_loss
                parents_loss = conf_p * (parents_pred - parents_nn_gt).abs() * parents_weights[:, None] + parent_conf_loss
            
            terms['joints_loss'] += joints_loss.mean()
            terms['parents_loss'] += parents_loss.mean()

            ##########################
            # Skin Loss
            ##########################
            if rep.success:
                # Encode skin features
                joint_skin_embeds_gt, vert_skin_embeds_gt = joint_skin_embeds_gt_list[i].detach(), vert_skin_embeds_gt_list[i].detach()
                # Calculate nearest vertices on GT to predicted mesh vertices
                mesh_verts = rep.vertices.detach()
                mesh_verts_gt = gt_mesh['vertices']

                # Joint mapping of: joints_gt, rep_skl.joints.detach()
                jmap = knn_points(rep_skl.joints_grouped.detach()[None], joints_gt[None], K=1).idx[0, :, 0]
                skin_gt = skin_gt[:, jmap]
                
                if 'cubvh' in kwargs:
                    bvh = kwargs['cubvh'][i].to(mesh_verts.device)
                    _, face_id, uvw = bvh.unsigned_distance(mesh_verts, return_uvw=True)
                    uvw = uvw.clamp(min=0.0)
                    uvw_sum = uvw.sum(dim=-1, keepdim=True).clamp_min(1e-3)
                    uvw = uvw / uvw_sum
                    face_id = gt_mesh['faces'][face_id]
                    skin_nn_gt = (skin_gt[face_id] * uvw[..., None]).sum(1)
                    vert_skin_embeds_gt_nn = (vert_skin_embeds_gt[face_id] * uvw[..., None]).sum(1)
                else:
                    _, vertex_nn_idx, _ = knn_points(mesh_verts[None], mesh_verts_gt[None], K=1, norm=2, return_nn=False)
                    vertex_nn_idx = vertex_nn_idx[0, :, 0]
                    # Use NN's skin as GT skin
                    skin_nn_gt = skin_gt[vertex_nn_idx]
                    vert_skin_embeds_gt_nn = vert_skin_embeds_gt[vertex_nn_idx]
                
                joint_skin_embeds_gt_nn = joint_skin_embeds_gt[joints_nn_idx]
                # Skin feature loss
                vert_skin_embeds_pred = rep.vertex_skin_feats
                joint_skin_embeds_pred = rep_skl.skin_feats
                skin_pred = rep_skl.skin_pred
                
                # Cache GT Skin Embeds
                if cache_gt_skin_embeds:
                    rep.vertex_skin_feats_gt = vert_skin_embeds_gt_nn
                    rep_skl.skin_feats_gt = joint_skin_embeds_gt_nn
                if cache_gt_nn_skin:
                    rep_skl.skin_nn_gt = skin_nn_gt
                
                if is_bad_skin:
                    # Ensure the parameters have gradients
                    vert_skin_term = l2_loss(vert_skin_embeds_pred, vert_skin_embeds_pred.detach())
                    joint_skin_term = l2_loss(joint_skin_embeds_pred, joint_skin_embeds_pred.detach())
                    terms['skin_feats_l2_loss_vert'] += vert_skin_term
                    terms['skin_feats_l2_loss_joints'] += joint_skin_term
                else:
                    vert_skin_term = l2_loss(vert_skin_embeds_pred, vert_skin_embeds_gt_nn)
                    joint_skin_diff = (joint_skin_embeds_pred - joint_skin_embeds_gt_nn) ** 2
                    if rep_skl.conf_skin is not None:
                        joint_skin_term = (rep_skl.conf_skin * joint_skin_diff).mean()
                    else:
                        joint_skin_term = joint_skin_diff.mean()
                    terms['skin_feats_l2_loss_vert'] += vert_skin_term
                    terms['skin_feats_l2_loss_joints'] += joint_skin_term
                    # KL Divergence for skin loss
                    skin_pred_f = skin_pred.float().clamp_min(1e-8)
                    skin_nn_gt_f = skin_nn_gt.float()
                    skin_nn_gt_f = skin_nn_gt_f / skin_nn_gt_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    skin_kl_loss = F.kl_div(skin_pred_f.log(), skin_nn_gt_f, reduction='batchmean')
                    terms['skin_kl_loss'] += skin_kl_loss
                    # Variance of joint's skin features from different vertices
                    terms['skin_feats_joints_var_loss'] += rep_skl.skin_feats_joints_var_loss

        for k in terms:
            terms[k] = (terms[k] / max(1, len([rep for rep in reps if rep.success]))) if 'skin' in k else (terms[k] / len(reps))

        return terms
    
    def training_losses(
        self,
        feats: SparseTensor,
        feats_skl: SparseTensor,

        joints,
        parents,
        skin,
        mesh: List[Dict],
        is_bad_skin,

        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        normal_map: torch.Tensor = None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar, z_skl, mean_skl, logvar_skl, joint_skin_embeds_gt, vert_skin_embeds_gt, joints_pos_gt = self.training_models['encoder'](feats, feats_skl, sample_posterior=True, return_raw=True, gt_joints=joints, gt_parents=parents, gt_skin=skin, gt_mesh=mesh, bvh_list=kwargs.get('cubvh', None))

        terms = edict(loss = 0.0, rec = 0.0)
        if self.latent_denoising:
            # Progressively increase the maximum time
            latent_time_max = min(1, (self.step / self.latent_achive_max_step)) * self.latent_time_max if self.latent_achive_max_step > 0 else self.latent_time_max
            encoder = getattr(self.training_models['encoder'], 'module', self.training_models['encoder'])
            latent_channels = encoder.latent_channels
            z_geo_feats, z_skin_feats = z.feats[:, :latent_channels], z.feats[:, latent_channels:]
            noise = torch.randn_like(z_skin_feats) * self.latent_denoising_gamma
            time = torch.rand([z.coords[:, 0].max() + 1]).to(z_skin_feats)[:, None]
            time = (1 - (1 - time).clip(min=1e-10).sqrt()) * latent_time_max
            time = time[z.coords[:, 0]]
            z_skin_feats = (1 - time) * z_skin_feats + time * noise
            z = z.replace(torch.cat([z_geo_feats, z_skin_feats], dim=1))
        else:
            terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
            terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        if self.latent_denoising_skl:
            # Progressively increase the maximum time
            latent_time_max_skl = min(1, (self.step / self.latent_achive_max_step_skl)) * self.latent_time_max_skl if self.latent_achive_max_step_skl > 0 else self.latent_time_max_skl
            noise_skl = torch.randn_like(z_skl.feats) * self.latent_denoising_gamma_skl
            time_skl = torch.rand([z_skl.coords[:, 0].max() + 1]).to(z_skl.feats)[:, None]
            time_skl = (1 - (1 - time_skl).clip(min=1e-10).sqrt()) * latent_time_max_skl
            time_skl = time_skl[z_skl.coords[:, 0]]
            z_skl = z_skl.replace((1 - time_skl) * z_skl.feats + time_skl * noise_skl)
        else:
            terms["kl_skl"] = 0.5 * torch.mean(mean_skl.pow(2) + logvar_skl.exp() - logvar_skl - 1)
            terms["loss"] = terms["loss"] + self.lambda_kl_skl * terms["kl_skl"]

        reps, reps_skl = self.training_models['decoder'](z, z_skl, joints, parents)
        self.renderer.rendering_options.resolution = image.shape[-1]

        terms['reg_loss'] = sum([rep.reg_loss for rep in reps]) / len(reps)
        terms['loss'] = terms['loss'] + terms['reg_loss']

        geo_terms = self.geometry_losses(reps, mesh, normal_map, extrinsics, intrinsics) if len(reps) > 0 else {'geo_loss': torch.tensor(0.0, device=self.device)}
        terms.update(geo_terms)
        terms['loss'] = terms['loss'] + terms['geo_loss']

        if reps_skl is not None:
            terms['reg_skl_loss'] = sum([rep_skl.reg_loss for rep_skl in reps_skl]) / len(reps_skl)
            terms['loss'] = terms['loss'] + terms['reg_skl_loss']

            skl_terms = self.skeleton_losses(joints, parents, skin, is_bad_skin, reps, reps_skl, mesh, joint_skin_embeds_gt, vert_skin_embeds_gt, **kwargs) if len(reps) > 0 else {'joints_loss': torch.tensor(0.0, device=self.device),
                'parents_loss': torch.tensor(0.0, device=self.device),
                'skin_feats_joints_var_loss': torch.tensor(0.0, device=self.device),
                'skin_kl_loss': torch.tensor(0.0, device=self.device),
                'skin_feats_l2_loss_vert': torch.tensor(0.0, device=self.device),
                'skin_feats_l2_loss_joints': torch.tensor(0.0, device=self.device),
            }
            terms.update(skl_terms)
            
            terms['loss'] = terms['loss'] + terms['joints_loss'] * self.lambda_joints
            terms['loss'] = terms['loss'] + terms['parents_loss'] * self.lambda_parents
            terms['loss'] = terms['loss'] + terms['skin_feats_joints_var_loss'] * self.lambda_skin_var
            terms['loss'] = terms['loss'] + terms['skin_kl_loss'] * self.lambda_skin_kl
            terms['loss'] = terms['loss'] + terms['skin_feats_l2_loss_vert'] * self.lambda_skin_feats_l2
            terms['loss'] = terms['loss'] + terms['skin_feats_l2_loss_joints'] * self.lambda_skin_feats_l2_skl
        
        if self.use_color:
            color_terms = self.color_losses(reps, image, alpha, extrinsics, intrinsics) if len(reps) > 0 else {'color_loss': torch.tensor(0.0, device=self.device)}
            terms.update(color_terms)
            terms['loss'] = terms['loss'] + terms['color_loss']
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        visualize_feats: bool = False,
        disturbance: float = 0.0,
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
        ret_dict = {}
        gt_images = []
        gt_normal_maps = []
        gt_meshes = []
        exts = []
        ints = []
        reps, reps_skl, instances = [], [], []
        joints_gt, parents_gt, skins_gt = [], [], []
        gt_skin_colors, pred_skin_colors, gt_supervised_colors = [], [], []
        error = {}
        incorrect_grouping_instances = []

        # Feature visualization for checking
        if visualize_feats:
            input_skin_feats_ss_imgs,      input_skin_feats_ssskl_imgs = [], []  # Visualize 1
            pred_skin_feats_verts_colors,  gt_skin_feats_verts_colors  = [], []  # Visualize 2
            pred_skin_feats_ssskl_imgs,    gt_skin_feats_ssskl_imgs    = [], []  # Visualize 3
            input_joint_pose_ssskl_imgs = []  # Visualize 4
        
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            for key in data:
                if type(data[key]) is list and hasattr(data[key][0], 'device'):
                    for j in range(len(data[key])):
                        data[key][j] = data[key][j].to(self.device)
            args = recursive_to_device(data, 'cuda')
            gt_images.append(args['image'] * args['alpha'][:, None])
            if self.use_color and 'normal_map' in data:
                gt_normal_maps.append(args['normal_map'])
            gt_meshes.extend(args['mesh'])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            
            z, z_skl, joint_skin_embeds_gt, vert_skin_embeds_gt, joints_pos_gt, x_skin, x_skl = self.models['encoder'](args['feats'], args['feats_skl'], sample_posterior=True, return_raw=False, return_skin_encoded=True, gt_joints=args['joints'], gt_parents=args['parents'], gt_skin=args['skin'], gt_mesh=args['mesh'], bvh_list=args.get('cubvh', None))
            if disturbance > 0.0:
                noise = torch.randn_like(z.feats)
                t = torch.rand([z.coords[:, 0].max() + 1]).to(z.feats)[:, None] * disturbance
                t = t[z.coords[:, 0]]
                z = z.replace((1 - t) * z.feats + t * noise)
                noise_skl = torch.randn_like(z_skl.feats)
                t_skl = torch.rand([z_skl.coords[:, 0].max() + 1]).to(z_skl.feats)[:, None] * disturbance
                t_skl = t_skl[z_skl.coords[:, 0]]
                z_skl = z_skl.replace((1 - t_skl) * z_skl.feats + t_skl * noise_skl)
            rep, rep_skl = self.models['decoder'](z, z_skl) if self.step >= 10000 else self.models['decoder'](z, z_skl, gt_joints=args['joints'], gt_parents=args['parents'])

            with torch.no_grad():
                skl_loss = self.skeleton_losses(args['joints'], args['parents'], args['skin'], args['is_bad_skin'], rep, rep_skl, args['mesh'], joint_skin_embeds_gt, vert_skin_embeds_gt, cache_gt_skin_embeds=True, cache_gt_nn_skin=True, **args)
            for k in skl_loss:
                if k not in error:
                    error[k] = []
                error[k].append(skl_loss[k])

            if visualize_feats:
                # Visualize 1 & 3
                coords_list, coords_skl_list, in_vert_skin_colors, in_skl_skin_colors, pred_skl_skin_colors, gt_skl_skin_colors, in_joint_pose_colors = [], [], [], [], [], [], []
                for j in range(x_skin.data.batch_size):
                    # Sparse Structure Coordinates
                    coords_list.append(x_skin[j].coords)
                    coords_skl_list.append(x_skl[j].coords)
                    # Input Skin Features
                    in_vert_skin_colors.append(get_transform(x_skin[j].feats)(x_skin[j].feats))
                    in_skl_skin_colors.append(get_transform(x_skl[j].feats)(x_skl[j].feats))
                    # Predicted Skeleton Skin Features
                    out_skl_skin_color_func = get_transform(rep_skl[j].skin_feats_gt)
                    pred_skl_skin_colors.append(out_skl_skin_color_func(rep_skl[j].skin_feats))
                    gt_skl_skin_colors.append(out_skl_skin_color_func(rep_skl[j].skin_feats_gt))
                    # Input JP Pose Features
                    in_joint_pose_colors.append(get_transform(joints_pos_gt[j])(joints_pos_gt[j]))
                input_skin_feats_ss_imgs.append(self.render_coords(coords_list, args['extrinsics'], args['intrinsics'], in_vert_skin_colors, self.models['encoder'].resolution))
                input_skin_feats_ssskl_imgs.append(self.render_coords(coords_skl_list, args['extrinsics'], args['intrinsics'], in_skl_skin_colors, self.models['encoder'].resolution))
                pred_skin_feats_ssskl_imgs.append(self.render_coords(coords_skl_list, args['extrinsics'], args['intrinsics'], pred_skl_skin_colors, self.models['encoder'].resolution))
                gt_skin_feats_ssskl_imgs.append(self.render_coords(coords_skl_list, args['extrinsics'], args['intrinsics'], gt_skl_skin_colors, self.models['encoder'].resolution))
                input_joint_pose_ssskl_imgs.append(self.render_coords(coords_skl_list, args['extrinsics'], args['intrinsics'], in_joint_pose_colors, self.models['encoder'].resolution))
                del coords_list, coords_skl_list, in_vert_skin_colors, in_skl_skin_colors, pred_skl_skin_colors, gt_skl_skin_colors, in_joint_pose_colors

            reps.extend(rep)
            reps_skl.extend(rep_skl)
            instances.extend(args['instance'])
            joints_gt.extend(args['joints'])
            parents_gt.extend(args['parents'])
            skins_gt.extend(args['skin'])
            
            skin_color_trans_funcs = [get_transform(skin) for skin in data['skin']]
            gt_skin_colors.extend([func(skin) for func, skin in zip(skin_color_trans_funcs, args['skin'])])
            joints_mapping = [knn_points(args['joints'][idx][None], rep_skl_.joints_grouped[None], K=1, norm=2, return_nn=False)[1][0, :, 0] for idx, rep_skl_ in enumerate(rep_skl)]
            pred_skin_mapped = [skin_pred_[:, jmap]  for jmap, skin_pred_ in zip(joints_mapping, [rs['skin_pred'] for rs in rep_skl])]
            pred_skin_colors.extend([func(skin) for func, skin in zip(skin_color_trans_funcs, pred_skin_mapped)])

            incorrect_grouping_instances.extend([ins for ins, jmap, rep_skl_ in zip(args['instance'], joints_mapping, rep_skl) if not (torch.bincount(jmap, minlength=rep_skl_.joints_grouped.shape[0]) == 1).all()])

            if visualize_feats:
                # Visualize 3
                skin_embed_color_trans_func = [get_transform(rep_.vertex_skin_feats_gt) for rep_ in rep]
                gt_skin_feats_verts_colors.extend([func(skin) for func, skin in zip(skin_embed_color_trans_func, [rep_.vertex_skin_feats_gt for rep_ in rep])])
                pred_skin_feats_verts_colors.extend([func(skin) for func, skin in zip(skin_embed_color_trans_func, [rep_.vertex_skin_feats for rep_ in rep])])

            gt_supervised_colors_list = []                
            for idx, (rep_, mesh, skin_gt, func) in enumerate(zip(rep, args['mesh'], args['skin'], skin_color_trans_funcs)):
                # Calculate nearest vertices on GT to predicted mesh vertices
                mesh_verts = rep_.vertices
                mesh_verts_gt = mesh['vertices']

                if 'cubvh' in args:
                    bvh = args['cubvh'][idx].to(mesh_verts.device)
                    _, face_id, uvw = bvh.unsigned_distance(mesh_verts, return_uvw=True)
                    uvw = uvw.clamp(min=0.0)
                    uvw_sum = uvw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    uvw = uvw / uvw_sum
                    face_id = mesh['faces'][face_id]
                    skin_nn_gt = (skin_gt[face_id] * uvw[..., None]).sum(1)
                else:
                    _, vertex_nn_idx, _ = knn_points(mesh_verts[None], mesh_verts_gt[None], K=1, norm=2, return_nn=False)
                    vertex_nn_idx = vertex_nn_idx[0, :, 0]
                    # Use NN's skin as GT skin
                    skin_nn_gt = skin_gt[vertex_nn_idx]
                
                gt_supervised_colors_list.append(func(skin_nn_gt))
            gt_supervised_colors.extend(gt_supervised_colors_list)

        ret_dict.update({f'incorrect_grouping_instances': {'value': incorrect_grouping_instances, 'type': 'text'}})

        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        if self.use_color and gt_normal_maps:
            gt_normal_maps = torch.cat(gt_normal_maps, dim=0)
            ret_dict.update({f'gt_normal_map': {'value': gt_normal_maps, 'type': 'image'}})

        if visualize_feats:
            # Visualize 1 & 3
            input_skin_feats_ss_imgs = torch.cat(input_skin_feats_ss_imgs, dim=0)
            input_skin_feats_ssskl_imgs = torch.cat(input_skin_feats_ssskl_imgs, dim=0)
            pred_skin_feats_ssskl_imgs = torch.cat(pred_skin_feats_ssskl_imgs, dim=0)
            gt_skin_feats_ssskl_imgs = torch.cat(gt_skin_feats_ssskl_imgs, dim=0)
            input_joint_pose_ssskl_imgs = torch.cat(input_joint_pose_ssskl_imgs, dim=0)
            ret_dict.update({f'input_ss_skin_embed': {'value': input_skin_feats_ss_imgs, 'type': 'image'}})
            ret_dict.update({f'input_skl_skin_embed': {'value': input_skin_feats_ssskl_imgs, 'type': 'image'}})
            ret_dict.update({f'output_pred_skin_embed_skl': {'value': pred_skin_feats_ssskl_imgs, 'type': 'image'}})
            ret_dict.update({f'output_gt_skin_embed_skl': {'value': gt_skin_feats_ssskl_imgs, 'type': 'image'}})
            ret_dict.update({f'input_joint_pose_ssskl': {'value': input_joint_pose_ssskl_imgs, 'type': 'image'}})

        ret_dict.update({'skin_error': {'value': torch.stack(error['skin_kl_loss']), 'type': 'scalar'}})
        ret_dict.update({'skin_feats_l2_error_vert': {'value': torch.stack(error['skin_feats_l2_loss_vert']), 'type': 'scalar'}})
        ret_dict.update({'skin_feats_l2_error_joints': {'value': torch.stack(error['skin_feats_l2_loss_joints']), 'type': 'scalar'}})
        ret_dict.update({'joints_error': {'value': torch.stack(error['joints_loss']), 'type': 'scalar'}})
        ret_dict.update({'parents_error': {'value': torch.stack(error['parents_loss']), 'type': 'scalar'}})
    
        # render single view
        # GT
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        return_types = ['normal']
        return_types.append('specified')
        gt_render_results = self._render_batch([
            MeshExtractResult(vertices=mesh['vertices'].to(self.device), faces=mesh['faces'].to(self.device))
            for mesh in gt_meshes
        ], exts, ints, return_types=return_types, specified_colors=gt_skin_colors)
        ret_dict.update({f'gt_normal': {'value': self._flip_normal(gt_render_results['normal'], exts, ints), 'type': 'image'}})
        if 'specified' in return_types:
            ret_dict.update({f'gt_skin_map': {'value': gt_render_results['specified'], 'type': 'image'}})
        # Pred
        return_types = ['normal']
        if self.use_color:
            return_types.append('color')
            if 'normal_map' in data:
                return_types.append('normal_map')
        return_types.append('specified')
        render_results = self._render_batch(reps, exts, ints, return_types=return_types, specified_colors=pred_skin_colors)
        gtsup_render_results = self._render_batch(reps, exts, ints, return_types=['specified'], specified_colors=gt_supervised_colors)
        ret_dict.update({f'rec_normal': {'value': render_results['normal'], 'type': 'image'}})
        if 'color' in return_types:
            ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})
        if 'normal_map' in return_types:
            ret_dict.update({f'rec_normal_map': {'value': render_results['normal_map'], 'type': 'image'}})
        if 'specified' in return_types:
            ret_dict.update({f'rec_skin_map': {'value': render_results['specified'], 'type': 'image'}})
        ret_dict.update({f'rec_skin_gtsup_map': {'value': gtsup_render_results['specified'], 'type': 'image'}})

        if visualize_feats:
            gt_skin_feats_verts_imgs = self._render_batch(reps, exts, ints, return_types=['specified'], specified_colors=gt_skin_feats_verts_colors)['specified']
            pred_skin_feats_verts_imgs = self._render_batch(reps, exts, ints, return_types=['specified'], specified_colors=pred_skin_feats_verts_colors)['specified']
            ret_dict.update({f'output_gt_skin_embed_verts': {'value': gt_skin_feats_verts_imgs, 'type': 'image'}})
            ret_dict.update({f'output_pred_skin_embed_verts': {'value': pred_skin_feats_verts_imgs, 'type': 'image'}})
    
        ############################################
        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        multiview_normals = []
        multiview_normal_maps = []
        multiview_skin_maps = []
        multiview_skin_gtsup_maps = []
        miltiview_images = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(len(reps), -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(len(reps), -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types, specified_colors=pred_skin_colors)
            gtsup_render_results = self._render_batch(reps, extrinsics, intrinsics, return_types=['specified'], specified_colors=gt_supervised_colors)
            multiview_normals.append(render_results['normal'])
            if 'color' in return_types:
                miltiview_images.append(render_results['color'])
            if 'normal_map' in return_types:
                multiview_normal_maps.append(render_results['normal_map'])
            if 'specified' in return_types:
                multiview_skin_maps.append(render_results['specified'])
            multiview_skin_gtsup_maps.append(gtsup_render_results['specified'])

        ## Concatenate views
        multiview_normals = torch.cat([
            torch.cat(multiview_normals[:2], dim=-2),
            torch.cat(multiview_normals[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'multiview_normal': {'value': multiview_normals, 'type': 'image'}})
        if 'color' in return_types:
            miltiview_images = torch.cat([
                torch.cat(miltiview_images[:2], dim=-2),
                torch.cat(miltiview_images[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_image': {'value': miltiview_images, 'type': 'image'}})
        if 'normal_map' in return_types:
            multiview_normal_maps = torch.cat([
                torch.cat(multiview_normal_maps[:2], dim=-2),
                torch.cat(multiview_normal_maps[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_normal_map': {'value': multiview_normal_maps, 'type': 'image'}})
        if 'specified' in return_types:
            # Concatenate specified colors
            multiview_specified_colors = torch.cat([
                torch.cat(multiview_skin_maps[:2], dim=-2),
                torch.cat(multiview_skin_maps[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_skin_map': {'value': multiview_specified_colors, 'type': 'image'}})
        multiview_skin_gtsup_maps = torch.cat([
            torch.cat(multiview_skin_gtsup_maps[:2], dim=-2),
            torch.cat(multiview_skin_gtsup_maps[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'multiview_skin_gtsup_map': {'value': multiview_skin_gtsup_maps, 'type': 'image'}})

        ## Skeletoned Meshes
        skeletoned_meshes = []
        skeletoned_meshes_gtskin = []
        
        if visualize_feats:
            gt_skin_feats_verts = [rep.vertex_skin_feats_gt for rep in reps]
            gt_skin_feats_skl = [rep_skl.skin_feats_gt for rep_skl in reps_skl]

            skin_preds_gt_skinfeats = self.models['decoder'].skinweight_forward(reps, reps_skl, return_skin_pred_only=True, skin_feats_verts_list=gt_skin_feats_verts, skin_feats_skl_list=gt_skin_feats_skl)
            skin_preds_gt_vert_skinfeats = self.models['decoder'].skinweight_forward(reps, reps_skl, return_skin_pred_only=True, skin_feats_verts_list=gt_skin_feats_verts)
            skin_preds_gt_skl_skinfeats = self.models['decoder'].skinweight_forward(reps, reps_skl, return_skin_pred_only=True, skin_feats_skl_list=gt_skin_feats_skl)

            skeletoned_meshes_gt_skinfeats = []
            skeletoned_meshes_gt_vert_skinfeats = []
            skeletoned_meshes_gt_skl_skinfeats = []

        joints_parents = []
        for i, rep_skl in enumerate(reps_skl):
            ##################################
            # Predicted mesh with Pred skeleton
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
            skeletoned_mesh_gtskin = dict(**skeletoned_mesh)
            skeletoned_mesh_gtskin['skin'] = rep_skl.skin_nn_gt
            skeletoned_meshes_gtskin.append(skeletoned_mesh_gtskin)

            if visualize_feats:
                # Check GT Skin Embeds
                skeletoned_mesh_gt_skinfeats = dict(**skeletoned_mesh)
                skeletoned_mesh_gt_skinfeats['skin'] = skin_preds_gt_skinfeats[i]
                skeletoned_meshes_gt_skinfeats.append(skeletoned_mesh_gt_skinfeats)
                skeletoned_mesh_gt_vert_skinfeats = dict(**skeletoned_mesh)
                skeletoned_mesh_gt_vert_skinfeats['skin'] = skin_preds_gt_vert_skinfeats[i]
                skeletoned_meshes_gt_vert_skinfeats.append(skeletoned_mesh_gt_vert_skinfeats)
                skeletoned_mesh_gt_skl_skinfeats = dict(**skeletoned_mesh)
                skeletoned_mesh_gt_skl_skinfeats['skin'] = skin_preds_gt_skl_skinfeats[i]
                skeletoned_meshes_gt_skl_skinfeats.append(skeletoned_mesh_gt_skl_skinfeats)

            ##################################
            # Joints and Parents Visualization
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
                if color_conf_j.max() > 1 + 1e-3 or color_conf_j.min() < 0 - 1e-3:
                    color_conf_j = (color_conf_j - color_conf_j.min()) / (color_conf_j.max() - color_conf_j.min() + 1e-8)
                color_conf_j = torch.cat([color_conf_j, torch.zeros_like(color_conf_j), 1-color_conf_j], dim=-1)
                joint_parent['joints_conf'] = torch.cat([joint_pred, color_conf_j], dim=-1)
            if rep_skl.conf_p is not None:
                color_conf_p = rep_skl.conf_p.expand(-1, 1)
                if color_conf_p.max() > 1 + 1e-3 or color_conf_p.min() < 0 - 1e-3:
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
            'skeletoned_meshes_gtskin': {'value': skeletoned_meshes_gtskin, 'type': 'skeletoned_mesh_list'},
            'joints_parents':    {'value': joints_parents,    'type': 'joints_parents'},
        })

        if visualize_feats:
            ret_dict.update({
                'skeletoned_meshes_gt_skinfeats':      {'value': skeletoned_meshes_gt_skinfeats,      'type': 'skeletoned_mesh_list'},
                'skeletoned_meshes_gt_vert_skinfeats': {'value': skeletoned_meshes_gt_vert_skinfeats, 'type': 'skeletoned_mesh_list'},
                'skeletoned_meshes_gt_skl_skinfeats':  {'value': skeletoned_meshes_gt_skl_skinfeats,  'type': 'skeletoned_mesh_list'},
            })

        return ret_dict
