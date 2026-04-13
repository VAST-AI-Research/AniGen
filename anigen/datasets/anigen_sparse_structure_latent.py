import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from .. import models
from ..utils.dist_utils import read_file_dist
import torch.nn.functional as F


class AniGenSparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = None,
        ss_dec_path: Optional[str] = '',
        ss_dec_ckpt: Optional[str] = 'final',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            # decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))  # Got stuck...
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, z_skl, batch_size=4):
        self._loading_ss_dec()
        ss = []
        ss_skl = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        if self.normalization_skl is not None:
            z_skl = z_skl * self.std_skl.to(z_skl.device) + self.mean_skl.to(z_skl.device)
        for i in range(0, z.shape[0], batch_size):
            z_, z_skl_ = z[i:i+batch_size], z_skl[i:i+batch_size]
            ss_, ss_skl_ = self.ss_dec(z_, z_skl_)
            ss.append(ss_)
            ss_skl.append(ss_skl_)
        ss = torch.cat(ss, dim=0)
        ss_skl = torch.cat(ss_skl, dim=0)
        self._delete_ss_dec()
        return ss, ss_skl

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict], x_0_skl: Optional[Union[torch.Tensor, dict]]=None, **kwargs):

        x_0_skl = x_0_skl if isinstance(x_0, torch.Tensor) else x_0['x_0_skl']
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0, x_0_skl = self.decode_latent(x_0.cuda(), x_0_skl.cuda())
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset =  0  # np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = np.linspace(-np.pi / 4, np.pi / 4, 4) # [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)

        x_0_skl = x_0_skl.cuda()
        for i in range(x_0_skl.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(x_0_skl[i, 0] > 0, as_tuple=False)
            resolution = x_0_skl.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images[i] = torch.cat([images[i], image], dim=2)

        return torch.stack(images)
       

class AniGenSparseStructureLatent(AniGenSparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        normalization_skl: Optional[dict] = None,
        pretrained_ss_dec: str = None,
        ss_dec_path: Optional[str] = '',
        ss_dec_ckpt: Optional[str] = 'final',
        **kwargs,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.normalization_skl = normalization_skl
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            **kwargs,
        )
        
        if self.normalization is not None:
            data = np.load(self.normalization)
            self.mean = torch.tensor(data['feats_mean'])
            self.std = torch.tensor(data['feats_std'])
        if self.normalization_skl is not None:
            data = np.load(self.normalization_skl)
            self.mean_skl = torch.tensor(data['feats_skl_mean'])
            self.std_skl = torch.tensor(data['feats_skl_std']).clip(min=1e-3)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        
        if 'is_bad_skeleton' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skeleton']]
        if 'is_bad_skin' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skin']]
        
        return metadata, stats
                
    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))
        z = torch.tensor(latent['mean']).float()
        z_skl = torch.tensor(latent['mean_skl']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        if self.normalization_skl is not None:
            z_skl = (z_skl - self.mean_skl) / self.std_skl

        pack = {
            'instance': instance,
            'x_0': z,
            'x_0_skl': z_skl,
        }
        return pack
    

class TextConditionedAniGenSparseStructureLatent(TextConditionedMixin, AniGenSparseStructureLatent):
    """
    Text-conditioned sparse structure dataset
    """
    pass


class ImageConditionedAniGenSparseStructureLatent(ImageConditionedMixin, AniGenSparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    pass
    