import os
from PIL import Image
import json
import numpy as np
import pandas as pd
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase


class AniGenSparseFeat2Skeleton(StandardDatasetBase):
    """
    SparseFeat2Render dataset.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        load_cubvh: bool = False,
        skl_dilation_iter: int = 0,
        skl_dilation_random_aug: bool = False,
        skl_dilation_random_aug_prob: float = 0.5,
        filter_bad_skin: bool = False,

        test_mode: bool = True,  # Test the model performance
        is_test: bool = False,  # Train or validation
        skin_accum_as_flow: bool = False,  # Accumulate skin weights from bottom to top as flow-by probability
        local_rank: int = 0,
        joint_merge_res: int = 64,
        **kwargs,
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.load_cubvh = load_cubvh
        self.skl_dilation_iter = skl_dilation_iter
        self.skl_dilation_random_aug = skl_dilation_random_aug
        self.skl_dilation_random_aug_prob = skl_dilation_random_aug_prob
        self.filter_bad_skin = filter_bad_skin

        self.test_mode = test_mode
        self.is_test = is_test
        self.skin_accum_as_flow = skin_accum_as_flow
        self.local_rank = local_rank
        self.joint_merge_res = joint_merge_res

        super().__init__(roots, **kwargs)
        self.is_bad_skin_list = self.metadata['is_bad_skin'].values
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'feature_{self.model}']]
        stats['With features'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)

        if 'is_bad_skeleton' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skeleton']]
        if self.filter_bad_skin and 'is_bad_skin' in metadata.columns:
            metadata = metadata[~metadata['is_bad_skin']]

        if self.test_mode:
            metadata = metadata[metadata['is_test']] if self.is_test else metadata[~metadata['is_test']]

        return metadata, stats

    def _get_image(self, root, instance):
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
    
    def _get_feat(self, root, instance):
        DATA_RESOLUTION = 64
        feats_path = os.path.join(root, 'features', self.model, f'{instance}.npz')
        feats_data = np.load(feats_path, allow_pickle=True)
        coords = torch.tensor(feats_data['indices']).int()
        feats = torch.tensor(feats_data['patchtokens']).float()

        position = utils3d.io.read_ply(os.path.join(root, 'voxels', f'{instance}_skeleton.ply'))[0]
        coords_skl = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous()
        ss_skl = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.long)
        ss_skl[0, coords_skl[:,0], coords_skl[:,1], coords_skl[:,2]] = 1
        ss_skl_ori = ss_skl.clone()
        if self.skl_dilation_random_aug or self.skl_dilation_iter > 0:
            size = max(0, self.skl_dilation_iter) * 2 + 1
            if self.skl_dilation_iter > 0:
                kernel = torch.ones(1, 1, size, size, size, dtype=torch.float32, device=ss_skl.device)
                ss_skl = torch.nn.functional.conv3d(ss_skl.float()[None], kernel, padding=self.skl_dilation_iter)
                ss_skl = (ss_skl > 0).long().squeeze(0)
                coords_skl = torch.nonzero(ss_skl[0], as_tuple=False).int()
            if self.skl_dilation_random_aug and np.random.rand() < self.skl_dilation_random_aug_prob:
                size_small, size_large = size - 2, size + 2
                kernel_large = torch.ones(1, 1, size_large, size_large, size_large, dtype=torch.float32, device=ss_skl_ori.device)
                ss_skl_large = torch.nn.functional.conv3d(ss_skl_ori.float()[None], kernel_large, padding=size_large//2)
                ss_skl_large = (ss_skl_large > 0).long().squeeze(0)
                if size_small > 1:
                    kernel_small = torch.ones(1, 1, size_small, size_small, size_small, dtype=torch.float32, device=ss_skl_ori.device)
                    ss_skl_small = torch.nn.functional.conv3d(ss_skl_ori.float()[None], kernel_small, padding=size_small//2)
                    ss_skl_small = (ss_skl_small > 0).long().squeeze(0)
                else:
                    ss_skl_small = torch.zeros_like(ss_skl)

                ss_skl_random_mask = torch.rand_like(ss_skl.float()) < 0.5
                ss_skl = ss_skl_small * ss_skl_random_mask.long() + ss_skl_large * (1 - ss_skl_random_mask.long())
                coords_skl = torch.nonzero(ss_skl[0], as_tuple=False).int()
        feats_skl = torch.zeros((coords_skl.shape[0], 0), dtype=torch.float32)
        
        if self.resolution != DATA_RESOLUTION:
            factor = DATA_RESOLUTION // self.resolution
            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce='mean'
            )
            coords_skl = coords_skl // factor
            coords_skl, idx = coords_skl.unique(return_inverse=True, dim=0)
            feats_skl = torch.scatter_reduce(
                torch.zeros(coords_skl.shape[0], feats_skl.shape[1], device=feats_skl.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats_skl.shape[1]),
                src=feats_skl,
                reduce='mean'
            )
        
        return {
            'coords': coords,
            'feats': feats,
            'coords_skl': coords_skl,
            'feats_skl': feats_skl,
        }

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        coords_skl = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
            coords_skl.append(torch.cat([torch.full((b['coords_skl'].shape[0], 1), i, dtype=torch.int32), b['coords_skl']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        pack['feats'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        coords_skl = torch.cat(coords_skl)
        feats_skl = torch.cat([b['feats_skl'] for b in batch])
        pack['feats_skl'] = SparseTensor(
            coords=coords_skl,
            feats=feats_skl,
        )

        pack['image'] = torch.stack([b['image'] for b in batch])
        pack['alpha'] = torch.stack([b['alpha'] for b in batch])
        pack['extrinsics'] = torch.stack([b['extrinsics'] for b in batch])
        pack['intrinsics'] = torch.stack([b['intrinsics'] for b in batch])
        
        pack['joints'] = [b['joints'] for b in batch]
        pack['parents'] = [b['parents'] for b in batch]
        pack['skin'] = [b['skin'] for b in batch]
        pack['is_bad_skin'] = [b['is_bad_skin'] for b in batch]
        
        # collate other data
        keys = [k for k in batch[0].keys() if k not in ['coords', 'feats', 'coords_skl', 'feats_skl', 'image', 'alpha', 'extrinsics', 'intrinsics', 'joints', 'parents', 'skin']]
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                pack[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], list):
                pack[k] = sum([b[k] for b in batch], [])
            else:
                pack[k] = [b[k] for b in batch]

        return pack
        
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
                bvh = torch.load(cubvh_path, weights_only=False)
                if isinstance(bvh, cuBVH):
                    bvh = bvh.to('cpu')
            else:
                device = torch.device(f"cuda:{self.local_rank}")
                bvh = cuBVH(mesh["vertices"], mesh["faces"], device=device)
                bvh = bvh.to('cpu')
                torch.save(bvh, cubvh_path)
            geo["cubvh"] = bvh
        return  geo

    def _get_skeleton(self, root, instance):
        skeleton_path = os.path.join(root, 'skeleton', instance, 'skeleton_voxelized.npz')
        skl_data = np.load(skeleton_path, allow_pickle=True)
        joints, parents, skin = skl_data['joints'], skl_data['parents'], skl_data['skin']
        parents[parents==None] = -1
        parents = np.array(parents, dtype=np.int32)

        skin[np.where(skl_data['skin'].max(axis=1)==0)[0], 0] = 1.0
        skin = skin / skin.sum(-1, keepdims=True)

        if self.skin_accum_as_flow:
            root_idx = np.where(parents == -1)[0][0]
            def sum_children(joint_idx, skin_weights):
                children = np.where(parents == joint_idx)[0]
                for child in children:
                    skin_weights[:, joint_idx] += sum_children(child, skin_weights)
                return skin_weights[:, joint_idx]
            sum_children(root_idx, skin)
            skin = np.clip(skin, 0, 1)
        
        is_bad_skin = self.metadata['is_bad_skin'][instance]

        return {
            'joints': torch.from_numpy(joints).float(),
            'parents': torch.from_numpy(parents).int(),
            'skin': torch.from_numpy(skin).float(),
            'is_bad_skin': is_bad_skin
        }

    def get_instance(self, root, instance):
        image = self._get_image(root, instance)
        feat = self._get_feat(root, instance)
        geo = self._get_geo(root, instance)
        skl = self._get_skeleton(root, instance)
        
        return {
            **image,
            **feat,
            **geo,
            **skl,
            'instance': instance,
        }
