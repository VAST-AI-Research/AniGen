import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import utils3d
from .components import StandardDatasetBase
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer


class AniGenSparseStructure(StandardDatasetBase):
    """
    Sparse structure dataset

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(self,
        roots,
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        skl_dilation_iter: int = 0,
        **kwargs,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.skl_dilation_iter = skl_dilation_iter
        self.value_range = (0, 1)

        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'voxelized']]
        stats['Voxelized'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats

    def get_ply_instance(self, root, instance, dilation_iter=None):
        if dilation_iter is None:
            dilation_iter = self.skl_dilation_iter

        position = utils3d.io.read_ply(os.path.join(root, 'voxels', f'{instance}.ply'))[0]
        coords = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous()
        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        if dilation_iter > 0:
            # 3D Dilation
            size = dilation_iter * 2 + 1
            kernel = torch.ones(1, 1, size, size, size, dtype=torch.float32, device=ss.device)
            ss = torch.nn.functional.conv3d(ss.float()[None], kernel, padding=dilation_iter)
            ss = (ss > 0).long().squeeze(0)
        return ss
    
    def get_instance(self, root, instance):
        ss = self.get_ply_instance(root, instance, dilation_iter=0)
        ss_skl = self.get_ply_instance(root, f'{instance}_skeleton', dilation_iter=self.skl_dilation_iter)
        return {'ss': ss, 'ss_skl': ss_skl, 'instance': instance}

    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        ss = ss if isinstance(ss, torch.Tensor) else ss['ss']
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = 0. # np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = np.linspace(-np.pi / 4, np.pi / 4, num=4)  # [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

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

        # Build each representation
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
       