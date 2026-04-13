from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from .anigen_base import AniGenSparseTransformerBase, FreqPositionalEmbedder
from pytorch3d.ops import knn_points
from .skin_models import SkinEncoder
    

def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, channels_out: int=None, mlp_ratio: float = 4.0):
        super().__init__()
        channels_out = channels if channels_out is None else channels_out
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AniGenSLatEncoder(AniGenSparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        
        model_channels: int,
        model_channels_skl: int,
        model_channels_skin: int,
        
        latent_channels: int,
        latent_channels_skl: int,
        latent_channels_vertskin: int,

        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,

        num_heads_skl: int = 32,
        num_heads_skin: int = 32,
        
        skl_pos_embed_freq: int = 10,
        skin_encoder_config: Optional[Dict[str, Any]] = {},
        encode_upsampled_skin_feat: bool = True,
        skin_ae_name: Optional[str] = "SkinAE",

        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        attn_mode_cross: Literal["full", "serialized", "windowed"] = "full",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,

        use_pretrain_branch: bool = True,
        freeze_pretrain_branch: bool = True,
        modules_to_freeze: Optional[List[str]] = ["input_layer", "blocks", "out_layer", "skin_encoder"],

        skin_cross_from_geo: bool = True,
        skl_cross_from_geo: bool = True,
        skin_skl_cross: bool = True,

        latent_denoising: bool = True,
        normalize_z: bool = True,
        normalize_scale: float = 1.0,

        jp_residual_fields: bool = False,
        jp_hyper_continuous: bool = False,
    ):
        self.use_pretrain_branch = use_pretrain_branch
        self.freeze_pretrain_branch = freeze_pretrain_branch
        self.skl_pos_embed_freq = skl_pos_embed_freq
        self.latent_denoising = latent_denoising
        self.normalize_latent = normalize_z and latent_denoising
        self.normalize_scale = normalize_scale
        self.jp_residual_fields = jp_residual_fields
        self.jp_hyper_continuous = jp_hyper_continuous
        
        super().__init__(
            in_channels=in_channels,
            in_channels_skl=model_channels_skl,
            in_channels_skin=model_channels_skin,
            model_channels=model_channels,
            model_channels_skl=model_channels_skl,
            model_channels_skin=model_channels_skin,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_heads_skl=num_heads_skl,
            num_heads_skin=num_heads_skin,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            attn_mode_cross=attn_mode_cross,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            skin_cross_from_geo=skin_cross_from_geo,
            skl_cross_from_geo=skl_cross_from_geo,
            skin_skl_cross=skin_skl_cross,
        )
        self.pretrain_class_name = ["AniGenElasticSLatEncoder", skin_ae_name]
        self.pretrain_ckpt_filter_prefix = {skin_ae_name: "skin_encoder"}
        self.resolution = resolution

        self.latent_channels = latent_channels
        self.latent_channels_skl = latent_channels_skl
        self.latent_channels_vertskin = latent_channels_vertskin

        skin_encoder_config['use_fp16'] = use_fp16
        self.skin_encoder = SkinEncoder(**skin_encoder_config)
        self.encode_upsampled_skin_feat = encode_upsampled_skin_feat
        self.in_layer_skin = FeedForwardNet(channels=self.skin_encoder.skin_feat_channels * (8 if encode_upsampled_skin_feat else 1), channels_out=model_channels_skin)

        self.pos_embedder_fourier = FreqPositionalEmbedder(in_dim=4 if self.jp_hyper_continuous else 3, max_freq_log2=self.skl_pos_embed_freq, num_freqs=self.skl_pos_embed_freq, include_input=True)
        self.root_embedding = nn.Parameter(torch.zeros(1, self.pos_embedder_fourier.out_dim))

        # Channel Balance
        self.in_layer_jp_skl = FeedForwardNet(channels=2 * self.pos_embedder_fourier.out_dim, channels_out=model_channels_skl//4)
        self.in_layer_skin_skl = FeedForwardNet(channels=self.skin_encoder.skin_feat_channels, channels_out=model_channels_skl-(model_channels_skl//4))

        self.out_layer = sp.SparseLinear(model_channels, 2 * latent_channels)
        if self.latent_denoising:
            self.out_layer_skl = sp.SparseLinear(model_channels_skl, latent_channels_skl)
            self.out_layer_vertskin = sp.SparseLinear(model_channels_skin, latent_channels_vertskin)
        else:
            self.out_layer_skl = sp.SparseLinear(model_channels_skl, 2 * latent_channels_skl)
            self.out_layer_vertskin = sp.SparseLinear(model_channels_skin, 2 * latent_channels_vertskin)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()
        
        if 'all' in modules_to_freeze:
            modules_to_freeze = list(set([k.split('.')[0] for k in self.state_dict().keys()]))
            print(f"\033[93mFreezing all modules: {modules_to_freeze}\033[0m")
        if self.use_pretrain_branch and self.freeze_pretrain_branch:
            for module in modules_to_freeze:
                if hasattr(self, module):
                    mod = getattr(self, module)
                    if isinstance(mod, nn.ModuleList):
                        for m in mod:
                            for name, param in m.named_parameters():
                                if 'lora' not in name:
                                    param.requires_grad = False
                    elif isinstance(mod, nn.Module):
                        for name, param in mod.named_parameters():
                            if 'lora' not in name:
                                param.requires_grad = False
                    elif isinstance(mod, torch.Tensor):
                        if mod.requires_grad:
                            mod.requires_grad = False

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def skeleton_embedding(self, x, x_skl, joints_list, parents_list, skin_list, gt_meshes, bvh_list=None):
        res = self.resolution
        feats_new = []
        feats_skl_new = []
        coords_new = []
        coords_skl_new = []

        joint_skin_embeds, vert_skin_embeds = self.skin_encoder(joints_list, parents_list, skin_list)
        joints_pos_list = []

        for i in range(len(joints_list)):
            parent_idx = parents_list[i].clone()
            
            coords_new.append(x[i].coords)
            coords_skl_new.append(x_skl[i].coords)
            coords_new[-1][:, 0] = i
            coords_skl_new[-1][:, 0] = i

            v_pos = (x[i].coords[:, 1:4] + 0.5) / res - 0.5
            v_pos_skl = (x_skl[i].coords[:, 1:4] + 0.5) / res - 0.5
            dist_nn_12, joints_nn_idx, _ = knn_points(v_pos_skl[None], joints_list[i][None], K=2, norm=2, return_nn=False)
            joints_nn_idx = joints_nn_idx[0, :, 0]

            # Skeleton positional embedding
            joints_pos  = joints_list[i][joints_nn_idx]             - (v_pos_skl if self.jp_residual_fields else 0)
            parents_pos = joints_list[i][parent_idx[joints_nn_idx]] - (v_pos_skl if self.jp_residual_fields else 0)
            if self.jp_hyper_continuous:
                factor      = (1 - (dist_nn_12[0, :, 0:1] / (dist_nn_12[0, :, 1:2] + 1e-8)).clamp(max=1.0))
                joints_pos  = torch.cat([joints_pos, factor], dim=-1)
                parents_pos = torch.cat([parents_pos, factor], dim=-1)
            joints_pos_embed  = self.pos_embedder_fourier(joints_pos)
            parents_pos_embed = self.pos_embedder_fourier(parents_pos)
            parents_pos_embed = torch.where(parent_idx[joints_nn_idx][:, None] == -1, self.root_embedding.expand_as(parents_pos_embed), parents_pos_embed)
            jp_pos_embed_nn   = torch.cat([joints_pos_embed, parents_pos_embed], dim=-1)
            jp_pos_embed_nn = self.in_layer_jp_skl(jp_pos_embed_nn)

            # Skeleton skin embedding
            j_skin_embed_nn = joint_skin_embeds[i][joints_nn_idx]
            j_skin_embed_nn = self.in_layer_skin_skl(j_skin_embed_nn)
            
            # Concatenate
            jp_skl_embed = torch.cat([jp_pos_embed_nn, j_skin_embed_nn], dim=-1)
            feats_skl_new.append(jp_skl_embed)

            if self.encode_upsampled_skin_feat:
                # Create 8 sub-voxel points
                offsets = torch.tensor([
                    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
                ], device=v_pos.device, dtype=v_pos.dtype) * (0.25 / res)
                query_pos = v_pos.unsqueeze(1) + offsets.unsqueeze(0) # (N, 8, 3)
                query_pos_flat = query_pos.view(-1, 3)
            else:
                query_pos_flat = v_pos

            if bvh_list is not None:
                bvh = bvh_list[i].to(v_pos.device)
                _, face_id, uvw = bvh.unsigned_distance(query_pos_flat, return_uvw=True)
                uvw = uvw.clamp(min=0.0)
                uvw_sum = uvw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                uvw = uvw / uvw_sum
                face_id = gt_meshes[i]['faces'][face_id]
                voxel_skin_embeds = (vert_skin_embeds[i][face_id] * uvw[..., None]).sum(1)
            else:
                gt_mesh_verts = gt_meshes[i]['vertices']
                _, mesh_nn_idx, _ = knn_points(query_pos_flat[None], gt_mesh_verts[None], K=1, norm=2, return_nn=False)
                mesh_nn_idx = mesh_nn_idx[0, :, 0]
                voxel_skin_embeds = vert_skin_embeds[i][mesh_nn_idx]
            
            voxel_skin_embeds = voxel_skin_embeds.view(v_pos.shape[0], -1)
            voxel_skin_embeds = self.in_layer_skin(voxel_skin_embeds)
            feats_new.append(voxel_skin_embeds)
            joints_pos_list.append(joints_pos)

        x_new     = sp.SparseTensor(coords=torch.cat(coords_new, dim=0), feats=torch.cat(feats_new, dim=0))
        x_skl_new = sp.SparseTensor(coords=torch.cat(coords_skl_new, dim=0), feats=torch.cat(feats_skl_new, dim=0))

        return x_new, x_skl_new, joint_skin_embeds, vert_skin_embeds, joints_pos_list
    
    def encode_sample(self, x: sp.SparseTensor, out_layer: sp.SparseLinear, sample_posterior: bool = True, latent_denoising: bool = False):
        x = x.type(torch.float32)
        x = x.replace(F.layer_norm(x.feats, x.feats.shape[-1:]))
        x = out_layer(x)
        if latent_denoising:
            if self.normalize_latent:
                x = x.replace(nn.functional.normalize(x.feats, dim=-1) * self.normalize_scale)
            mean, logvar = x.feats, torch.zeros_like(x.feats)
        else:
            mean, logvar = x.feats.chunk(2, dim=-1)
        if sample_posterior and not latent_denoising:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = x.replace(z)
        if latent_denoising:
            mean = mean.detach()
        return z, mean, logvar

    def forward(self, x: sp.SparseTensor, x_skl: sp.SparseTensor, sample_posterior=True, return_raw=False, return_skin_encoded=False, **kwargs):
        x_skin, x_skl, joint_skin_embeds, vert_skin_embeds, joints_pos = self.skeleton_embedding(x, x_skl, kwargs.get('gt_joints'), kwargs.get('gt_parents'), kwargs.get('gt_skin'), kwargs.get('gt_mesh'), kwargs.get('bvh_list', None))
        h, h_skl, h_skin = super().forward(x, x_skl, x_skin)

        z, mean, logvar = self.encode_sample(h, self.out_layer, sample_posterior, latent_denoising=False)
        z_skl, mean_skl, logvar_skl = self.encode_sample(h_skl, self.out_layer_skl, sample_posterior, latent_denoising=self.latent_denoising)
        z_skin, mean_skin, logvar_skin = self.encode_sample(h_skin, self.out_layer_vertskin, sample_posterior, latent_denoising=self.latent_denoising)

        z = z.replace(torch.cat([z.feats, z_skin.feats], dim=-1))
        mean, logvar = torch.cat([mean, mean_skin], dim=-1), torch.cat([logvar, logvar_skin], dim=-1)
        
        if not return_skin_encoded:
            # Ordinary return without skin encoded features
            if return_raw:
                return z, mean, logvar, z_skl, mean_skl, logvar_skl, joint_skin_embeds, vert_skin_embeds, joints_pos
            else:
                return z, z_skl, joint_skin_embeds, vert_skin_embeds, joints_pos
        else:
            # Return skin encoded features as well for checking
            if return_raw:
                return z, mean, logvar, z_skl, mean_skl, logvar_skl, joint_skin_embeds, vert_skin_embeds, joints_pos, x_skin, x_skl
            else:
                return z, z_skl, joint_skin_embeds, vert_skin_embeds, joints_pos, x_skin, x_skl

    def encode_skin(self, joints_list: List[torch.Tensor], parents_list: List[torch.Tensor], skin_list: List[torch.Tensor]=None):
        joint_skin_embeds, vert_skin_embeds = self.skin_encoder(joints_list, parents_list, skin_list)
        return joint_skin_embeds, vert_skin_embeds


class AniGenElasticSLatEncoder(SparseTransformerElasticMixin, AniGenSLatEncoder):
    """
    SLat VAE encoder with elastic memory management.
    Used for training with low VRAM.
    """
