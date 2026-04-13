from typing import *
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import sys
import trimesh
from easydict import EasyDict as edict

from anigen import models
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..utils.model_utils import load_model_from_path, load_decoder
from ..utils.image_utils import load_dsine, preprocess_image, encode_image
from ..utils.skin_utils import repair_skeleton_parents, smooth_skin_weights_on_mesh, filter_skinning_weights
from ..utils.export_utils import convert_to_glb_from_data, _extract_vertex_rgb, visualize_skeleton_as_mesh
from ..utils.postprocessing_utils import (
    postprocess_mesh,
    barycentric_transfer_attributes,
    parametrize_mesh,
    bake_texture,
)
from ..utils.general_utils import _keep_largest_connected_component_3d


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class AnigenImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Anigen image-to-3D models.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        ss_config: edict = None,
        slat_config: edict = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.ss_config = ss_config
        self.slat_config = slat_config
        # self.device is handled by the base Pipeline class property

    @staticmethod
    def from_pretrained(
        ss_flow_path: str = 'ckpts/anigen/ss_flow_solo',
        slat_flow_path: str = 'ckpts/anigen/slat_flow_control',
        device: str = 'cuda',
        use_ema: bool = False
    ) -> "AnigenImageTo3DPipeline":
        """
        Load pretrained models from paths.
        """
        print("Loading models...")
        
        # Image Cond Model (DINOv2)
        dinov2_model = torch.hub.load('./ckpts/dinov2', 'dinov2_vitl14_reg', pretrained=True, source='local')
        dinov2_model.to(device).eval()

        # DSINE Model
        print("Loading DSINE...")
        dsine_model = load_dsine(device)

        # SLat Flow Model
        slat_model, slat_config = load_model_from_path(slat_flow_path, model_name_in_config='denoiser', device=device, use_ema=use_ema)
        
        # SLat Decoder
        slat_dec_path = slat_config.dataset.args.get('slat_dec_path')
        slat_dec_ckpt = slat_config.dataset.args.get('slat_dec_ckpt')
        print(f"Loading SLat Decoder from {slat_dec_path}...")
        slat_decoder = load_decoder(slat_dec_path, slat_dec_ckpt, device)

        # SS Flow Model
        ss_model, ss_config = load_model_from_path(ss_flow_path, model_name_in_config='denoiser', device=device, use_ema=use_ema)

        # SS Decoder
        ss_dec_path = ss_config.dataset.args.get('ss_dec_path')
        ss_dec_ckpt = ss_config.dataset.args.get('ss_dec_ckpt')
        print(f"Loading SS Decoder from {ss_dec_path}...")
        ss_decoder = load_decoder(ss_dec_path, ss_dec_ckpt, device)

        models_dict = {
            'image_cond_model': dinov2_model,
            'dsine': dsine_model,
            'slat_flow_model': slat_model,
            'slat_decoder': slat_decoder,
            'ss_flow_model': ss_model,
            'ss_decoder': ss_decoder,
        }

        pipeline = AnigenImageTo3DPipeline(models_dict, ss_config, slat_config)
        return pipeline

    def load_ss_flow_model(self, ss_flow_path: str, device: str = 'cuda', use_ema: bool = False):
        """
        Hot-swap the SS flow model (and its config) without reloading decoders or shared models.
        """
        print(f"Loading SS flow model from {ss_flow_path}...")
        old_model = self.models.pop('ss_flow_model', None)
        if old_model is not None:
            if hasattr(old_model, 'cpu'):
                old_model.cpu()
            del old_model
            _cuda_cleanup()
        ss_model, ss_config = load_model_from_path(ss_flow_path, model_name_in_config='denoiser', device=device, use_ema=use_ema)
        ss_model.to(device).eval()
        self.models['ss_flow_model'] = ss_model
        self.ss_config = ss_config

    def load_slat_flow_model(self, slat_flow_path: str, device: str = 'cuda', use_ema: bool = False):
        """
        Hot-swap the SLAT flow model (and its config) without reloading decoders or shared models.
        """
        print(f"Loading SLAT flow model from {slat_flow_path}...")
        old_model = self.models.pop('slat_flow_model', None)
        if old_model is not None:
            if hasattr(old_model, 'cpu'):
                old_model.cpu()
            del old_model
            _cuda_cleanup()
        slat_model, slat_config = load_model_from_path(slat_flow_path, model_name_in_config='denoiser', device=device, use_ema=use_ema)
        slat_model.to(device).eval()
        self.models['slat_flow_model'] = slat_model
        self.slat_config = slat_config

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Preprocess the input image using DSINE model for normal estimation.
        Returns (processed_rgb, processed_normal).
        """
        image, normal_image = preprocess_image(image, self.models['dsine'], str(self.device))
        return image, normal_image

    def get_cond(self, image: Image.Image, normal_image: Image.Image) -> dict:
        """
        Get conditioning for SS and SLat models.
        """
        cond_rgb = encode_image(image, self.models['image_cond_model'], self.device)
        cond_normal = encode_image(normal_image, self.models['image_cond_model'], self.device)

        # Conditioning tensors for flow models
        normal_tensor = torch.from_numpy(np.array(normal_image)).float() / 255.0
        normal_tensor = normal_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        rgb_tensor = torch.from_numpy(np.array(image.convert('RGB'))).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        cond_dict_ss = {
            'cond': cond_normal,
            'neg_cond': torch.zeros_like(cond_rgb),
            'normal': normal_tensor
        }
        
        cond_dict_slat_rgb = {
            'cond': cond_rgb,
            'neg_cond': torch.zeros_like(cond_rgb),
            'normal': rgb_tensor
        }

        return cond_dict_ss, cond_dict_slat_rgb

    def sample_sparse_structure(
        self, 
        cond_dict_ss: dict,
        strength: float = 7.5,
        steps: int = 50,
        skl_mainland_only: bool = True,
        progress_callback=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sparse structure (SS) and skeleton (SS_skl).
        Returns (coords, coords_skl).
        """
        ss_model = self.models['ss_flow_model']
        ss_decoder = self.models['ss_decoder']
        
        print("Sampling Sparse Structure...")
        ss_sampler = samplers.AniGenFlowEulerCfgSampler(sigma_min=1e-5)
        reso = ss_model.resolution
        
        noise = torch.randn(1, ss_model.in_channels, reso, reso, reso).to(self.device)
        if ss_model.z_is_global:
             noise = torch.randn(1, ss_model.global_token_num, ss_model.in_channels).to(self.device)

        noise_skl = torch.randn(1, ss_model.in_channels_skl, reso, reso, reso).to(self.device)
        if ss_model.z_skl_is_global:
            noise_skl = torch.randn(1, ss_model.global_token_num_skl, ss_model.in_channels_skl).to(self.device)

        z_s_out = ss_sampler.sample(
            ss_model,
            noise,
            noise_skl,
            **cond_dict_ss,
            steps=steps,
            cfg_strength=strength,
            verbose=True,
            progress_callback=progress_callback,
        )

        z_s = z_s_out.samples
        z_s_skl = z_s_out.samples_skl

        decoded_ss, decoded_ss_skl = ss_decoder(z_s, z_s_skl)

        if skl_mainland_only:
            bsz, ch, d, h, w = decoded_ss_skl.shape
            for b in range(bsz):
                occ_3d = (decoded_ss_skl[b] > 0).any(dim=0).detach().cpu().numpy()
                if not np.any(occ_3d):
                    continue
                mainland_3d = _keep_largest_connected_component_3d(occ_3d)
                mainland_t = torch.from_numpy(mainland_3d).to(device=decoded_ss_skl.device)
                mainland_cd = mainland_t.unsqueeze(0).expand(ch, -1, -1, -1)
                decoded_ss_skl[b] = torch.where(
                    mainland_cd,
                    decoded_ss_skl[b],
                    torch.full_like(decoded_ss_skl[b], -1e9),
                )
        coords = torch.argwhere(decoded_ss > 0)[:, [0, 2, 3, 4]].int()
        coords_skl = torch.argwhere(decoded_ss_skl > 0)[:, [0, 2, 3, 4]].int()
        
        return coords, coords_skl, decoded_ss, decoded_ss_skl

    def sample_slat(
        self,
        cond_dict_slat: dict,
        coords: torch.Tensor,
        coords_skl: torch.Tensor,
        strength: float = 3.0,
        steps: int = 50,
        joint_density: int = 1,
        progress_callback=None,
    ) -> Tuple[sp.SparseTensor, sp.SparseTensor]:
        """
        Sample Structured Latent (SLat) features.
        """
        slat_model = self.models['slat_flow_model']
        
        print("Sampling Structured Latent...")

        # Auto-detect geodesic smooth noise settings from config
        gsn_enabled = False
        gsn_iters = 0
        gsn_alpha = 0.7
        if hasattr(self, 'slat_config') and self.slat_config is not None:
            trainer_args = getattr(getattr(self.slat_config, 'trainer', None), 'args', None)
            if trainer_args is not None:
                gsn_enabled = bool(getattr(trainer_args, 'geodesic_smooth_noise', False))
                gsn_iters = int(getattr(trainer_args, 'geodesic_smooth_noise_iters', 0))
                gsn_alpha = float(getattr(trainer_args, 'geodesic_smooth_noise_alpha', 0.7))

        slat_sampler = samplers.AniGenFlowEulerCfgSampler(
            sigma_min=1e-5,
            geodesic_smooth_noise=gsn_enabled,
            geodesic_smooth_noise_iters=gsn_iters,
            geodesic_smooth_noise_alpha=gsn_alpha,
        )

        noise_slat = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], slat_model.in_channels + slat_model.in_channels_vert_skin).to(self.device),
            coords=coords,
        )
        noise_skl = sp.SparseTensor(
            feats=torch.randn(coords_skl.shape[0], slat_model.in_channels_skl).to(self.device),
            coords=coords_skl,
        )
        
        # Prepare conditioning
        cond = cond_dict_slat.copy()
        
        # Check for joint density conditioning support
        use_joint_num_cond = bool(getattr(slat_model, 'use_joint_num_cond', False))
        
        if use_joint_num_cond:
            joints_num = {0: 0, 1: 10, 2: 15, 3: 25, 4: 35}.get(joint_density, 10)
            cond['joints_num'] = joints_num
            cond['neg_joints_num'] = 0
        
        out = slat_sampler.sample(
            slat_model,
            noise_slat,
            noise_skl,
            **cond,
            steps=steps,
            cfg_strength=strength,
            verbose=True,
            progress_callback=progress_callback,
        )
        
        slat = out.samples
        slat_skl = out.samples_skl
        
        # Normalization
        if 'dataset' in self.slat_config and 'args' in self.slat_config.dataset and 'normalization' in self.slat_config.dataset.args:
            norm_stats = self.slat_config.dataset.args.normalization
            
            def denormalize(tensor, mean, std):
                if tensor is None: return None
                mean = torch.tensor(mean).to(tensor.device)
                std = torch.tensor(std).to(tensor.device)
                return tensor * std + mean
            
            if 'slat' in norm_stats:
                slat = slat.replace(feats=denormalize(slat.feats, norm_stats['slat']['mean'], norm_stats['slat']['std']))
            elif 'mean' in norm_stats and 'std' in norm_stats:
                slat = slat.replace(feats=denormalize(slat.feats, norm_stats['mean'], norm_stats['std']))
            
            if 'slat_skl' in norm_stats:
                 slat_skl = slat_skl.replace(feats=denormalize(slat_skl.feats, norm_stats['slat_skl']['mean'], norm_stats['slat_skl']['std']))
            elif 'slat_skel' in norm_stats:
                 slat_skl = slat_skl.replace(feats=denormalize(slat_skl.feats, norm_stats['slat_skel']['mean'], norm_stats['slat_skel']['std']))
            elif 'mean_skl' in norm_stats and 'std_skl' in norm_stats:
                 slat_skl = slat_skl.replace(feats=denormalize(slat_skl.feats, norm_stats['mean_skl'], norm_stats['std_skl']))
        
        return slat, slat_skl

    def decode_slat(self, slat: sp.SparseTensor, slat_skl: sp.SparseTensor):
        slat_decoder = self.models['slat_decoder']
        meshes, skeletons = slat_decoder(slat, slat_skl)
        return meshes[0], skeletons[0]

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        seed: int = 42,
        cfg_scale_ss: float = 7.5,
        cfg_scale_slat: float = 3.0,
        ss_steps: int = 25,
        slat_steps: int = 25,
        joints_density: int = 1,
        simplify_ratio: float = 0.95,
        fill_holes: bool = True,
        no_smooth_skin_weights: bool = False,
        no_filter_skin_weights: bool = False,
        smooth_skin_weights_iters: int = 100,
        smooth_skin_weights_alpha: float = 1.0,
        texture_size: int = 1024,
        output_glb: str = None,
        ss_progress_callback=None,
        slat_progress_callback=None,
        postprocess_progress_callback=None,
    ) -> dict:
        
        def _pp_progress(frac, desc):
            if postprocess_progress_callback is not None:
                postprocess_progress_callback(frac, desc)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Preprocess
        processed_image, processed_normal = self.preprocess_image(image)
        
        # Get conditioning
        cond_dict_ss, cond_dict_slat_rgb = self.get_cond(processed_image, processed_normal)
        
        # Sample SS
        coords, coords_skl, _, _ = self.sample_sparse_structure(
            cond_dict_ss, 
            strength=cfg_scale_ss, 
            steps=ss_steps,
            progress_callback=ss_progress_callback,
        )
        del cond_dict_ss
        _cuda_cleanup()

        # Sample SLat
        slat, slat_skl = self.sample_slat(
            cond_dict_slat_rgb, 
            coords, 
            coords_skl, 
            strength=cfg_scale_slat, 
            steps=slat_steps,
            joint_density=joints_density,
            progress_callback=slat_progress_callback,
        )
        del cond_dict_slat_rgb
        coords_cpu = coords.cpu()
        coords_skl_cpu = coords_skl.cpu()
        del coords, coords_skl
        _cuda_cleanup()
        
        # Decode
        mesh_result, skeleton_result = self.decode_slat(slat, slat_skl)
        del slat, slat_skl
        _cuda_cleanup()
        
        # ---------- Post-processing ----------
        _pp_progress(0.0, "Post-processing...")

        joints = skeleton_result.joints_grouped.cpu().numpy()
        parents = skeleton_result.parents_grouped.cpu().numpy().astype(np.int32)
        parents = repair_skeleton_parents(joints=joints, parents=parents, verbose=False).astype(np.int32)
        
        skin_weights = skeleton_result.skin_pred.cpu().numpy()
        vertex_colors = _extract_vertex_rgb(getattr(mesh_result, 'vertex_attrs', None))

        orig_vertices = mesh_result.vertices.cpu().numpy()
        orig_faces = mesh_result.faces.cpu().numpy()
        del skeleton_result
        _cuda_cleanup()

        _pp_progress(0.05, "Simplification...")

        new_vertices, new_faces = postprocess_mesh(
            orig_vertices, orig_faces,
            simplify=(simplify_ratio > 0),
            simplify_ratio=simplify_ratio,
            fill_holes=fill_holes,
            verbose=True,
        )

        # Transfer skin weights via barycentric interpolation
        if new_vertices.shape[0] != orig_vertices.shape[0]:
            orig_mesh = trimesh.Trimesh(vertices=orig_vertices, faces=orig_faces, process=False)
            skin_weights = barycentric_transfer_attributes(orig_mesh, skin_weights, new_vertices)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        if not no_filter_skin_weights:
            skin_weights = filter_skinning_weights(
                mesh,
                skin_weights,
                joints,
                parents,
            )

        if not no_smooth_skin_weights:
            skin_weights = smooth_skin_weights_on_mesh(
                mesh,
                skin_weights,
                iterations=smooth_skin_weights_iters,
                alpha=smooth_skin_weights_alpha,
        )
        
        # UV parametrize and bake texture via multiview rendering (mesh-based teacher)
        texture_image = None
        if texture_size > 0:
            _pp_progress(0.20, "Baking texture...")

            uv_vertices, uv_faces, uvs, vmapping = parametrize_mesh(
                new_vertices, new_faces
            )
            skin_weights = skin_weights[vmapping]

            # Teacher: dense mesh with vertex colors rendered by MeshRenderer
            from ..utils.render_utils import render_multiview

            observations, extrinsics_mv, intrinsics_mv = render_multiview(
                mesh_result, resolution=1024, nviews=100,
            )
            masks = [np.any(obs > 0, axis=-1) for obs in observations]
            extrinsics_np = [e.cpu().numpy() for e in extrinsics_mv]
            intrinsics_np = [i.cpu().numpy() for i in intrinsics_mv]
            del extrinsics_mv, intrinsics_mv, mesh_result
            _cuda_cleanup()

            with torch.enable_grad():
                texture = bake_texture(
                    uv_vertices, uv_faces, uvs,
                    observations, masks, extrinsics_np, intrinsics_np,
                    texture_size=texture_size, mode='opt',
                    lambda_tv=0.01,
                    verbose=True,
                )
            texture_image = texture
            del observations, masks, extrinsics_np, intrinsics_np
            _cuda_cleanup()

            mesh = trimesh.Trimesh(
                vertices=uv_vertices,
                faces=uv_faces,
                visual=trimesh.visual.TextureVisuals(uv=uvs),
                process=False,
            )
        else:
            del mesh_result
            _cuda_cleanup()
        
        _pp_progress(0.90, "Exporting...")

        skeleton_mesh = visualize_skeleton_as_mesh(joints, parents)
        
        ret = {
            'mesh': mesh,
            'skeleton_mesh': skeleton_mesh,
            'joints': joints,
            'parents': parents,
            'skin_weights': skin_weights,
            'vertex_colors': vertex_colors,
            'texture_image': texture_image,
            'processed_image': processed_image,
            'processed_normal': processed_normal,
            'coords': coords_cpu,
            'coords_skl': coords_skl_cpu,
        }

        if output_glb:
            convert_to_glb_from_data(
                mesh,
                joints,
                parents,
                skin_weights,
                output_glb,
                vertex_colors=vertex_colors,
                texture_image=texture_image,
            )
            if skeleton_mesh is not None and len(skeleton_mesh.vertices) > 0:
                skeleton_glb_path = os.path.join(os.path.dirname(output_glb), "skeleton.glb")
                skeleton_mesh.export(skeleton_glb_path)

        _pp_progress(1.0, "Done!")
        _cuda_cleanup()
            
        return ret
