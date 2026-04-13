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


class AniGenSkinAETrainer(BasicTrainer):
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
        lambda_kl_skin: float = 1.0,
        lambda_l1_skin: float = 0.,
        lambda_l2_skin: float = 0.,
        num_skin_samples: Optional[int] = None,

        regularizations: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_kl_skin = lambda_kl_skin
        self.lambda_l1_skin = lambda_l1_skin
        self.lambda_l2_skin = lambda_l2_skin
        self.num_skin_samples = num_skin_samples
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
            coords = coords_list[i]
            representation.position = coords.float() / ss_resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(ss_resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position if colors_overwrite_list is None else colors_overwrite_list[i])
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        return torch.stack(images)
    
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

    def _sample_barycentric_coords(self, num_samples: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(num_samples, device=device)
        v = torch.rand(num_samples, device=device)
        sqrt_u = torch.sqrt(u)
        bary_coords = torch.stack([1 - sqrt_u, sqrt_u * (1 - v), sqrt_u * v], dim=-1)
        return bary_coords

    def _sample_skin_from_surface(self, mesh: Dict[str, torch.Tensor], skin: torch.Tensor, num_samples: int) -> torch.Tensor:
        # Sample surface points proportional to triangle area and barycentrically interpolate skin weights.
        device = skin.device
        vertices = mesh['vertices'].to(device)
        faces = mesh['faces'].to(device)
        if faces.dtype != torch.long:
            faces = faces.long()
        face_vertices = vertices[faces]
        vec0 = face_vertices[:, 1] - face_vertices[:, 0]
        vec1 = face_vertices[:, 2] - face_vertices[:, 0]
        cross_prod = torch.cross(vec0, vec1, dim=-1)
        face_areas = torch.linalg.norm(cross_prod, dim=-1) * 0.5
        face_areas = torch.clamp(face_areas, min=1e-12)
        if not torch.isfinite(face_areas).all() or face_areas.sum() <= 0:
            face_areas = torch.ones_like(face_areas)
        probs = face_areas / face_areas.sum()
        face_indices = torch.multinomial(probs, num_samples, replacement=True)
        sampled_faces = faces[face_indices]
        bary_coords = self._sample_barycentric_coords(num_samples, device=device)
        skin_vertices = skin[sampled_faces]
        sampled_skin = torch.sum(bary_coords.unsqueeze(-1) * skin_vertices, dim=1)
        return sampled_skin

    def _prepare_surface_skin_samples(self, mesh_list: List[Dict[str, torch.Tensor]], skin_list: List[torch.Tensor]) -> List[torch.Tensor]:
        sampled_skins: List[torch.Tensor] = []
        for mesh, skin in zip(mesh_list, skin_list):
            num_samples = skin.shape[0] if self.num_skin_samples is None else self.num_skin_samples
            sampled_skins.append(self._sample_skin_from_surface(mesh, skin, num_samples))
        return sampled_skins

    def skeleton_losses(
        self,
        skin_pred_list,
        skin_gt_list,
    ):
        skin_kl_loss = 0.0
        skin_l1_loss = 0.0
        skin_l2_loss = 0.0
        for i in range(len(skin_pred_list)):
            skin_pred = skin_pred_list[i]
            skin_gt = skin_gt_list[i]
            skin_kl_loss = skin_kl_loss + F.kl_div(torch.log(skin_pred + 1e-10), skin_gt, reduce="batchmean")
            if self.lambda_l1_skin > 0:
                skin_l1_loss = skin_l1_loss + l1_loss(skin_pred, skin_gt)
            if self.lambda_l2_skin > 0:
                skin_l2_loss = skin_l2_loss + l2_loss(skin_pred, skin_gt)
        skin_kl_loss = skin_kl_loss / len(skin_pred_list)
        skin_l1_loss = skin_l1_loss / len(skin_pred_list)
        skin_l2_loss = skin_l2_loss / len(skin_pred_list)
        return {'skin_kl_loss': skin_kl_loss, 'skin_l1_loss': skin_l1_loss, 'skin_l2_loss': skin_l2_loss}

    def training_losses(
        self,
        feats: SparseTensor,
        feats_skl: SparseTensor,

        joints,
        parents,
        skin,
        mesh: List[Dict],

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
            mesh: The list of mesh dicts with 'vertices' and 'faces'.
            skin: The list of skin tensors of shape [num_vertices x num_joints].

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        sampled_skin = self._prepare_surface_skin_samples(mesh, skin)
        skin_pred_list, joint_skin_embeds, vert_skin_embeds = self.training_models['model'](joints_list=joints, parents_list=parents, skin_list=sampled_skin)
        self.renderer.rendering_options.resolution = image.shape[-1]
        terms = edict(loss = 0.0)
        skin_loss = self.skeleton_losses(skin_pred_list, sampled_skin)
        terms.loss = terms.loss + self.lambda_kl_skin * skin_loss['skin_kl_loss'] + self.lambda_l1_skin * skin_loss['skin_l1_loss'] + self.lambda_l2_skin * skin_loss['skin_l2_loss']
        terms.update(skin_loss)
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
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
        instances = []
        gt_skin_colors, pred_skin_colors = [], []
        error = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            for key in data:
                if type(data[key]) is list and hasattr(data[key][0], 'device'):
                    for j in range(len(data[key])):
                        data[key][j] = data[key][j].to(self.device)
            args = recursive_to_device(data, 'cuda')
            gt_images.append(args['image'] * args['alpha'][:, None])
            gt_meshes.extend(args['mesh'])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])

            skin_pred_list, joint_skin_embeds, vert_skin_embeds = self.training_models['model'](joints_list=args['joints'], parents_list=args['parents'], skin_list=args['skin'])

            with torch.no_grad():
                skl_loss = self.skeleton_losses(skin_pred_list, args['skin'])
            for k in skl_loss:
                if k not in error:
                    error[k] = []
                error[k].append(skl_loss[k])

            instances.extend(args['instance'])
            
            skin_color_trans_funcs = [get_transform(skin) for skin in data['skin']]
            gt_skin_colors.extend([func(skin) for func, skin in zip(skin_color_trans_funcs, args['skin'])])
            pred_skin_colors.extend([func(skin) for func, skin in zip(skin_color_trans_funcs, skin_pred_list)])

        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        ret_dict.update({'skin_error': {'value': torch.stack(error['skin_kl_loss']), 'type': 'scalar'}})
        
        # render single view
        # GT
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        return_types = ['specified']
        gt_render_results = self._render_batch([
            MeshExtractResult(vertices=mesh['vertices'].to(self.device), faces=mesh['faces'].to(self.device))
            for mesh in gt_meshes
        ], exts, ints, return_types=return_types, specified_colors=gt_skin_colors)
        ret_dict.update({f'gt_skin_map': {'value': gt_render_results['specified'], 'type': 'image'}})
        # Pred
        render_results = self._render_batch([
            MeshExtractResult(vertices=mesh['vertices'].to(self.device), faces=mesh['faces'].to(self.device))
            for mesh in gt_meshes
        ], exts, ints, return_types=return_types, specified_colors=pred_skin_colors)
        ret_dict.update({f'rec_skin_map': {'value': render_results['specified'], 'type': 'image'}})
        return ret_dict
