import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.dist_utils import read_file_dist
from ..utils.data_utils import load_balanced_group_indices
import copy
import torch.nn.functional as F


class AniGenSLatVisMixin:
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = None,
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        load_cubvh: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        self.load_cubvh = load_cubvh
        
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            # decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.slat_dec
        self.slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, z_skl, gt_joints=None, gt_parents=None, batch_size=4, gt_reps=None, gt_reps_skl=None):
        self._loading_slat_dec()
        reps = []
        reps_skl = []
        if gt_reps is not None:
            skins_gt_ssskin = []
        if gt_reps_skl is not None:
            skins_gt_sklskin = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
            z_skl = z_skl * self.std_skl.to(z.device) + self.mean_skl.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            gt_j, gt_p = None if gt_joints is None else gt_joints[i:i+batch_size], None if gt_parents is None else gt_parents[i:i+batch_size]
            z_, z_skl_ = z[i:i+batch_size], z_skl[i:i+batch_size]
            rep, rep_skl = self.slat_dec(z_, z_skl_, gt_joints=gt_j, gt_parents=gt_p)
            reps.append(rep)
            reps_skl.append(rep_skl)
            if gt_reps is not None:
                skins_gt_ssskin.append(self.slat_dec.skinweight_forward(gt_reps[i:i+batch_size], rep_skl, gt_joints=gt_j, gt_parents=gt_p, return_skin_pred_only=True))
            if gt_reps_skl is not None:
                skins_gt_sklskin.append(self.slat_dec.skinweight_forward(rep, gt_reps_skl[i:i+batch_size], gt_joints=gt_j, gt_parents=gt_p, return_skin_pred_only=True))
        reps = sum(reps, [])
        reps_skl = sum(reps_skl, [])
        self._delete_slat_dec()
        to_return = (reps, reps_skl)
        if gt_reps is not None:
            skins_gt_ssskin = sum(skins_gt_ssskin, [])
            to_return += (skins_gt_ssskin,)
        if gt_reps_skl is not None:
            skins_gt_sklskin = sum(skins_gt_sklskin, [])
            to_return += (skins_gt_sklskin,)
        return to_return
    
class AniGenSLat(AniGenSLatVisMixin, StandardDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        use_joint_num_cond: bool = False,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = None,
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        local_rank: int = 0,
        **kwargs,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.use_joint_num_cond = use_joint_num_cond
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.local_rank = local_rank
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
            **kwargs,
        )

        self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in self.instances]
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
            self.mean_skl = torch.tensor(self.normalization['mean_skl']).reshape(1, -1)
            self.std_skl = torch.tensor(self.normalization['std_skl']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        # metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        # stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        
        if 'is_bad_skeleton' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skeleton']]
        if 'is_bad_skin' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skin']]
        
        return metadata, stats

    @torch.no_grad()
    def visualize_sample(self, data: dict):
        return {}
        x_0 = data['x_0']
        x_0_skl = data['x_0_skl']
        reps, reps_skl = self.decode_latent(x_0.cuda(), x_0_skl.cuda(), data['joints'])
        
        # Build camera
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

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        images = torch.stack(images)
            
        return images

    def _get_skeleton(self, root, instance):
        skeleton_path = os.path.join(root, 'skeleton', instance, 'skeleton_voxelized.npz')
        skl_data = np.load(skeleton_path, allow_pickle=True)
        joints, parents, skin = skl_data['joints'], skl_data['parents'], skl_data['skin']
        parents[parents==None] = -1
        parents = np.array(parents, dtype=np.int32)
        ret = {
            'joints': torch.from_numpy(joints).float(),
            'parents': torch.from_numpy(parents).int(),
            'skin': torch.from_numpy(skin).float(),
        }
        if self.use_joint_num_cond:
            ret['joints_num'] = int(joints.shape[0])
        return ret
        
    def _get_geo(self, root, instance):
        skeleton_path = os.path.join(root, 'skeleton', instance, 'skeleton_voxelized.npz')
        skl_data = np.load(skeleton_path, allow_pickle=True)
        verts, face = np.array(skl_data['vertices'], dtype=np.float32), skl_data['faces']
        mesh = {
            "vertices" : torch.from_numpy(verts),
            "faces" : torch.from_numpy(face),
        }
        geo = {"mesh": mesh}
        if self.load_cubvh:
            from cubvh import cuBVH
            torch.cuda.set_device(self.local_rank)
            cubvh_path = os.path.join(root, 'skeleton', instance, 'cubvh.pth')
            if os.path.exists(cubvh_path):
                cubvh = torch.load(cubvh_path, weights_only=False)
            else:
                cubvh = cuBVH(mesh["vertices"], mesh["faces"])
                torch.save(cubvh, cubvh_path)
            geo["cubvh"] = cubvh
        return  geo

    def get_instance(self, root, instance):
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        coords_skl = torch.tensor(data['coords_skl']).int()
        feats_skl = torch.tensor(data['feats_skl']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
            feats_skl = (feats_skl - self.mean_skl) / self.std_skl
        return {
            'coords': coords,
            'feats': feats,
            'coords_skl': coords_skl,
            'feats_skl': feats_skl,
            'instance': instance,
            **self._get_skeleton(root, instance),
            **self._get_geo(root, instance),
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            coords_skl = []
            feats_skl = []
            layout = []
            layout_skl = []
            start = 0
            start_skl = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                coords_skl.append(torch.cat([torch.full((b['coords_skl'].shape[0], 1), i, dtype=torch.int32), b['coords_skl']], dim=-1))
                feats_skl.append(b['feats_skl'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                layout_skl.append(slice(start_skl, start_skl + b['coords_skl'].shape[0]))
                start += b['coords'].shape[0]
                start_skl += b['coords_skl'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            coords_skl = torch.cat(coords_skl)
            feats_skl = torch.cat(feats_skl)
            pack['x_0_skl'] = SparseTensor(
                coords=coords_skl,
                feats=feats_skl,
            )
            pack['x_0_skl']._shape = torch.Size([len(group), *sub_batch[0]['feats_skl'].shape[1:]])
            pack['x_0_skl'].register_spatial_cache('layout', layout_skl)
            
            pack['joints'] = [b['joints'] for b in sub_batch]
            pack['parents'] = [b['parents'] for b in sub_batch]
            pack['skin'] = [b['skin'] for b in sub_batch]
            if 'joints_num' in sub_batch[0]:
                pack['joints_num'] = torch.tensor([b['joints_num'] for b in sub_batch], dtype=torch.long)
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats', 'coords_skl', 'feats_skl', 'joints', 'parents', 'skin', 'joints_num']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs
        
    
class TextConditionedSLat(TextConditionedMixin, AniGenSLat):
    """
    Text conditioned structured latent dataset
    """
    pass


class AniGenImageConditionedSLat(ImageConditionedMixin, AniGenSLat):
    """
    Image conditioned structured latent dataset
    """
    pass
