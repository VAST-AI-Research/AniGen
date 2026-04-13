from typing import *
import contextlib
import torch
import torch.nn as nn
from ...modules.sparse.transformer import SparseTransformerMultiContextCrossBlock, SparseTransformerBlock
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from ...representations import MeshExtractResult
from ...representations.mesh import AniGenSparseFeatures2Mesh, AniGenSklFeatures2Skeleton
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from pytorch3d.ops import knn_points
from .anigen_base import AniGenSparseTransformerBase, FreqPositionalEmbedder
from .skin_models import SKIN_MODEL_DICT
import torch.nn.functional as F
from ...representations.skeleton.grouping import GROUPING_STRATEGIES


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        sub_divide: bool = True,
        conv_as_residual: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2 if sub_divide else resolution
        self.out_channels = out_channels or channels
        self.sub_divide = sub_divide
        self.conv_as_residual = conv_as_residual

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide() if sub_divide else nn.Identity()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels and not self.conv_as_residual:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
SparseConv3d
        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h


class SparseDownsampleWithCache(nn.Module):
    """SparseDownsample that stores upsample caches under a unique suffix.

    This avoids cache-key collisions when stacking multiple down/up stages.
    """
    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]], cache_suffix: str):
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.cache_suffix = cache_suffix
        self._down = sp.SparseDownsample(self.factor)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        out = self._down(x)

        dim = out.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * dim
        k_coords = f'upsample_{factor}_coords'
        k_layout = f'upsample_{factor}_layout'
        k_idx = f'upsample_{factor}_idx'
        coords = out.get_spatial_cache(k_coords)
        layout = out.get_spatial_cache(k_layout)
        idx = out.get_spatial_cache(k_idx)
        if any(v is None for v in [coords, layout, idx]):
            raise ValueError('Downsample cache not found after SparseDownsample.')

        # spconv expects int32 indices; SparseDownsample produces int64 coords.
        if out.coords.dtype != torch.int32:
            out = sp.SparseTensor(
                out.feats,
                out.coords.to(torch.int32),
                out.shape,
                out.layout,
                scale=out._scale,
                spatial_cache=out._spatial_cache,
            )

        out.register_spatial_cache(f'upsample_{factor}_{self.cache_suffix}_coords', coords)
        out.register_spatial_cache(f'upsample_{factor}_{self.cache_suffix}_layout', layout)
        out.register_spatial_cache(f'upsample_{factor}_{self.cache_suffix}_idx', idx)
        # Remove unsuffixed keys to prevent later stages overwriting them.
        try:
            del out._spatial_cache[k_coords]
            del out._spatial_cache[k_layout]
            del out._spatial_cache[k_idx]
        except Exception:
            pass
        return out


class SparseUpsampleWithCache(nn.Module):
    """SparseUpsample that reads upsample caches under a unique suffix."""
    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]], cache_suffix: str):
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.cache_suffix = cache_suffix

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        dim = x.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * dim
        new_coords = x.get_spatial_cache(f'upsample_{factor}_{self.cache_suffix}_coords')
        new_layout = x.get_spatial_cache(f'upsample_{factor}_{self.cache_suffix}_layout')
        idx = x.get_spatial_cache(f'upsample_{factor}_{self.cache_suffix}_idx')
        if any(v is None for v in [new_coords, new_layout, idx]):
            raise ValueError('Upsample cache not found. Must be paired with SparseDownsampleWithCache.')
        if new_coords.dtype != torch.int32:
            new_coords = new_coords.to(torch.int32)
        new_feats = x.feats[idx]
        out = sp.SparseTensor(new_feats, new_coords, x.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(x._scale, factor)])
        out._spatial_cache = x._spatial_cache
        return out


class SparseSkinUNetNLevel(nn.Module):
    """A simple N-down/N-up sparse UNet for local smoothing.

    Note: `SparseSubdivideBlock3d` uses `resolution` only to name spconv `indice_key`s.
    We must provide distinct (and stage-appropriate) values per hierarchy to avoid
    rulebook collisions across different coordinate sets.
    """
    def __init__(self, channels: int, base_resolution: int, num_groups: int = 32, num_levels: int = 3):
        super().__init__()

        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        self.channels = channels
        self.base_resolution = int(base_resolution)
        self.num_groups = num_groups
        self.num_levels = int(num_levels)

        def res_block(resolution: int):
            return SparseSubdivideBlock3d(
                channels=channels,
                resolution=resolution,
                out_channels=channels,
                sub_divide=False,
                conv_as_residual=True,
                num_groups=num_groups,
            )

        # resolutions[i] corresponds to the i-th encoder stage (before downsample)
        resolutions: List[int] = [max(1, self.base_resolution // (2 ** i)) for i in range(self.num_levels)]
        bottom_resolution = max(1, self.base_resolution // (2 ** self.num_levels))

        self.enc = nn.ModuleList([res_block(r) for r in resolutions])
        self.down = nn.ModuleList([SparseDownsampleWithCache(2, f'unet{i}') for i in range(self.num_levels)])
        self.mid = res_block(bottom_resolution)

        # Decoder blocks operate at the same resolutions as encoder blocks.
        self.up = nn.ModuleList([SparseUpsampleWithCache(2, f'unet{i}') for i in range(self.num_levels)])
        self.fuse = nn.ModuleList([sp.SparseLinear(channels * 2, channels) for _ in range(self.num_levels)])
        self.dec = nn.ModuleList([res_block(r) for r in resolutions])

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        in_dtype = x.feats.dtype
        if x.coords.dtype != torch.int32:
            x = sp.SparseTensor(
                x.feats,
                x.coords.to(torch.int32),
                x.shape,
                x.layout,
                scale=x._scale,
                spatial_cache=x._spatial_cache,
            )

        # spconv implicit_gemm has a runtime tuner that can fail for some sparse
        # rulebooks under AMP + fp16/bf16. Running UNet convs in fp32 avoids that.
        if hasattr(torch, 'autocast'):
            autocast_ctx = torch.autocast(device_type=x.device.type, enabled=False)
        else:
            # Older torch fallback
            autocast_ctx = torch.cuda.amp.autocast(enabled=False) if x.device.type == 'cuda' else contextlib.nullcontext()

        with autocast_ctx:
            x_fp32 = x if x.feats.dtype == torch.float32 else x.replace(x.feats.float())

            skips: List[sp.SparseTensor] = []
            h = x_fp32
            for i in range(self.num_levels):
                s = self.enc[i](h)
                skips.append(s)
                h = self.down[i](s)

            h = self.mid(h)

            for i in reversed(range(self.num_levels)):
                h_up = self.up[i](h)
                s = skips[i]
                h = self.fuse[i](h_up.replace(torch.cat([h_up.feats, s.feats], dim=-1)))
                h = self.dec[i](h)

            u0 = h

        if in_dtype != u0.feats.dtype:
            u0 = u0.replace(u0.feats.to(dtype=in_dtype))
        return u0


class AniGenSLatMeshDecoder(AniGenSparseTransformerBase):
    def __init__(
        self,
        resolution: int,
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
        
        skin_cross_from_groupped: bool = False,
        h_skin_unet_num_levels: int = 4,

        skin_decoder_config: Optional[Dict[str, Any]] = {},
        
        upsample_skl: bool = False,
        skl_defined_on_center: bool = True,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        attn_mode_cross: Literal["full", "serialized", "windowed"] = "full",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,

        use_pretrain_branch: bool = True,
        freeze_pretrain_branch: bool = True,
        modules_to_freeze: Optional[List[str]] = ["blocks", "upsample", "out_layer", "skin_decoder"],

        skin_cross_from_geo: bool = False,
        skl_cross_from_geo: bool = False,
        skin_skl_cross: bool = False,
        skin_ae_name: str = "SkinAE",

        normalize_z: bool = False,
        normalize_scale: float = 1.0,
        
        jp_residual_fields: bool = True,
        jp_hyper_continuous: bool = True,

        grouping_strategy: Literal["mean_shift", "threshold"] = "mean_shift",

        vertex_skin_feat_interp_sparse: bool = False,
        vertex_skin_feat_interp_nearest: bool = False,
        vertex_skin_feat_interp_use_deformed_grid: bool = False,
        vertex_skin_feat_interp_trilinear: bool = False,
        flexicube_disable_deform: bool = False,
        vertex_skin_feat_nodeform_trilinear: bool = False,
    ):
        super().__init__(
            in_channels=latent_channels,
            in_channels_skl=latent_channels_skl,
            in_channels_skin=latent_channels_vertskin,
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
        self.pretrain_class_name = ["AniGenElasticSLatMeshDecoder", skin_ae_name]
        self.pretrain_ckpt_filter_prefix = {skin_ae_name: "skin_decoder"}
        self.latent_channels = latent_channels
        self.latent_channels_skl = latent_channels_skl
        self.latent_channels_vertskin = latent_channels_vertskin
        self.jp_residual_fields = jp_residual_fields
        self.jp_hyper_continuous = jp_hyper_continuous
        self.grouping_func = GROUPING_STRATEGIES[grouping_strategy]
        self.skin_cross_from_groupped = skin_cross_from_groupped

        self.normalize_z = normalize_z
        self.normalize_scale = normalize_scale

        skin_decoder_config['use_fp16'] = use_fp16
        self.skin_decoder = SKIN_MODEL_DICT[skin_decoder_config.pop('model_type')](**skin_decoder_config)
        self.skin_feat_channels = self.skin_decoder.skin_feat_channels

        # Optional local smoothing UNet on h_skin (independent of grouped cross-attn).
        # If `h_skin_unet_num_levels < 0`, UNet is disabled.
        self.h_skin_unet_num_levels = int(h_skin_unet_num_levels)
        if self.h_skin_unet_num_levels >= 1:
            self.h_skin_unet = SparseSkinUNetNLevel(
                model_channels_skin,
                base_resolution=resolution,
                num_levels=self.h_skin_unet_num_levels,
            )
        else:
            self.h_skin_unet = None

        if self.skin_cross_from_groupped:
            # Trainable parent feature for root joints (where parent_idx < 0).
            self.root_parent_feat = nn.Parameter(torch.zeros(self.skin_feat_channels))

            # Joint feature preprocessing: [joint_skin, fourier(joint_xyz), parent_skin] -> proj -> self-attn
            self.joints_pos_embedder = FreqPositionalEmbedder(
                in_dim=3,
                include_input=True,
                max_freq_log2=6,
                num_freqs=6,
                log_sampling=True,
            )
            joints_pe_dim = self.joints_pos_embedder.out_dim
            joints_in_dim = self.skin_feat_channels + joints_pe_dim + self.skin_feat_channels
            self.joints_ctx_channels = model_channels_skin
            self.joints_in_proj = nn.Sequential(
                nn.Linear(joints_in_dim, self.joints_ctx_channels, bias=True),
                nn.SiLU(),
                nn.LayerNorm(self.joints_ctx_channels, elementwise_affine=True),
            )
            self.joints_self_attn = nn.ModuleList([
                SparseTransformerBlock(
                    self.joints_ctx_channels,
                    num_heads=num_heads_skin,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    window_size=None,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=False,
                    qk_rms_norm=self.qk_rms_norm,
                    ln_affine=True,
                ) for _ in range(4)
            ])

            # Coordinate PE for h_skin before cross-attn: coords in [-1, 1] -> Fourier PE -> proj(C), concat, fuse back to C.
            self.h_skin_coord_embedder = FreqPositionalEmbedder(
                in_dim=3,
                include_input=True,
                max_freq_log2=6,
                num_freqs=6,
                log_sampling=True,
            )
            h_skin_pe_dim = self.h_skin_coord_embedder.out_dim
            self.h_skin_coord_proj = nn.Linear(h_skin_pe_dim, model_channels_skin, bias=True)
            self.h_skin_coord_fuse = sp.SparseLinear(model_channels_skin * 2, model_channels_skin)

            self.skin_cross_groupped_net = SparseTransformerMultiContextCrossBlock(
                model_channels_skin,
                # Context includes processed joint tokens + raw joint skin feats (skip connection).
                ctx_channels=[self.joints_ctx_channels + self.skin_feat_channels],
                num_heads=num_heads_skin,
                mlp_ratio=self.mlp_ratio,
                attn_mode="full",
                attn_mode_cross="full",
                cross_attn_cache_suffix='_skin_cross_from_groupped',
            )

        self.resolution = resolution
        self.use_pretrain_branch = use_pretrain_branch
        self.freeze_pretrain_branch = freeze_pretrain_branch
        self.upsample_skl = upsample_skl
        self.rep_config = representation_config
        self.mesh_extractor = AniGenSparseFeatures2Mesh(
            res=self.resolution*4, 
            use_color=self.rep_config.get('use_color', False), 
            skin_feat_channels=self.skin_feat_channels, 
            predict_skin=True,
            vertex_skin_feat_interp_sparse=vertex_skin_feat_interp_sparse,
            vertex_skin_feat_interp_nearest=vertex_skin_feat_interp_nearest,
            vertex_skin_feat_interp_use_deformed_grid=vertex_skin_feat_interp_use_deformed_grid,
            vertex_skin_feat_interp_trilinear=vertex_skin_feat_interp_trilinear,
            flexicube_disable_deform=flexicube_disable_deform,
            vertex_skin_feat_nodeform_trilinear=vertex_skin_feat_nodeform_trilinear,
        )
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        upsample_skin_blocks = []
        upsample_skin_blocks.extend([
            SparseSubdivideBlock3d(
                channels=model_channels_skin,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])

        self.upsample_skin_net = nn.ModuleList(upsample_skin_blocks)
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)
        self.out_layer_skin = sp.SparseLinear(model_channels // 8, self.skin_feat_channels*8)
        self.out_layer_skl_skin = sp.SparseLinear(model_channels // 8 if upsample_skl else model_channels_skl, self.skin_feat_channels if skl_defined_on_center else self.skin_feat_channels * 8)
        self.use_conf_jp   = self.rep_config.get('use_conf_jp', False) or self.jp_hyper_continuous
        self.use_conf_skin = self.rep_config.get('use_conf_skin', False)

        res_skl = self.resolution * 4 if self.upsample_skl else self.resolution
        self.skeleton_extractor = AniGenSklFeatures2Skeleton(skin_feat_channels=self.skin_feat_channels, device=self.device, res=res_skl, use_conf_jp=self.use_conf_jp, use_conf_skin=self.use_conf_skin, predict_skin=True, defined_on_center=skl_defined_on_center, jp_hyper_continuous=self.jp_hyper_continuous, jp_residual_fields=self.jp_residual_fields)
        
        self.out_channels_skl = self.skeleton_extractor.feats_channels
        if self.upsample_skl:
            self.upsample_skl_net = nn.ModuleList([
                SparseSubdivideBlock3d(
                    channels=model_channels_skl,
                    resolution=resolution,
                    out_channels=model_channels // 4
                ),
                SparseSubdivideBlock3d(
                    channels=model_channels // 4,
                    resolution=resolution * 2,
                    out_channels=model_channels // 8
                )
            ])
            self.out_layer_skl = sp.SparseLinear(model_channels // 8, self.out_channels_skl)
        else:
            self.out_layer_skl = sp.SparseLinear(model_channels_skl, self.out_channels_skl)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()
        
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
        scale = 1e-4
        # Kaiming initialization for output layers (better for ReLU/SiLU-like activations)
        nn.init.kaiming_normal_(self.out_layer.weight, mode='fan_in', nonlinearity='relu')
        self.out_layer.weight.data.mul_(scale)
        nn.init.constant_(self.out_layer.bias, 0)

        nn.init.kaiming_normal_(self.out_layer_skl.weight, mode='fan_in', nonlinearity='relu')
        self.out_layer_skl.weight.data.mul_(scale)
        nn.init.constant_(self.out_layer_skl.bias, 0)
        
        # Initialize skin layer:
        self.skin_decoder.initialize_weights()
        nn.init.kaiming_normal_(self.out_layer_skin.weight, mode='fan_in', nonlinearity='relu')
        self.out_layer_skin.weight.data.mul_(scale)
        nn.init.constant_(self.out_layer_skin.bias, 0)
        
        nn.init.kaiming_normal_(self.out_layer_skl_skin.weight, mode='fan_in', nonlinearity='relu')
        self.out_layer_skl_skin.weight.data.mul_(scale)
        nn.init.constant_(self.out_layer_skl_skin.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)
        self.upsample_skin_net.apply(convert_module_to_f16)
        if self.upsample_skl:
            self.upsample_skl_net.apply(convert_module_to_f16)
        if self.skin_cross_from_groupped:
            # Joint preprocessing and cross-attn should match model dtype.
            self.root_parent_feat.data = self.root_parent_feat.data.half()
            self.joints_in_proj.apply(convert_module_to_f16)
            self.joints_self_attn.apply(convert_module_to_f16)

            # `convert_module_to_f16` doesn't include `nn.LayerNorm`, so cast LN params explicitly.
            for _m in self.joints_in_proj.modules():
                if isinstance(_m, nn.LayerNorm):
                    if _m.weight is not None:
                        _m.weight.data = _m.weight.data.half()
                    if _m.bias is not None:
                        _m.bias.data = _m.bias.data.half()

            # IMPORTANT: `SparseTransformerBlock` uses `LayerNorm32` which internally
            # normalizes in fp32 (`x.float()`), so its parameters must stay fp32.
            for _m in self.joints_self_attn.modules():
                if isinstance(_m, nn.LayerNorm):
                    if _m.weight is not None:
                        _m.weight.data = _m.weight.data.float()
                    if _m.bias is not None:
                        _m.bias.data = _m.bias.data.float()

            self.skin_cross_groupped_net.apply(convert_module_to_f16)
            self.h_skin_coord_proj.apply(convert_module_to_f16)
            self.h_skin_coord_fuse.apply(convert_module_to_f16)

        # UNet is executed in fp32 (see `SparseSkinUNetNLevel.forward`), so keep its
        # weights in fp32 to avoid dtype mismatches inside spconv.
        if self.h_skin_unet is not None:
            self.h_skin_unet.apply(convert_module_to_f32)
        self.skin_decoder.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)
        self.upsample_skin_net.apply(convert_module_to_f32)
        if self.upsample_skl:
            self.upsample_skl_net.apply(convert_module_to_f32)
        if self.skin_cross_from_groupped:
            self.root_parent_feat.data = self.root_parent_feat.data.float()
            self.joints_in_proj.apply(convert_module_to_f32)
            self.joints_self_attn.apply(convert_module_to_f32)

            for _m in self.joints_in_proj.modules():
                if isinstance(_m, nn.LayerNorm):
                    if _m.weight is not None:
                        _m.weight.data = _m.weight.data.float()
                    if _m.bias is not None:
                        _m.bias.data = _m.bias.data.float()
            for _m in self.joints_self_attn.modules():
                if isinstance(_m, nn.LayerNorm):
                    if _m.weight is not None:
                        _m.weight.data = _m.weight.data.float()
                    if _m.bias is not None:
                        _m.bias.data = _m.bias.data.float()

            self.skin_cross_groupped_net.apply(convert_module_to_f32)
            self.h_skin_coord_proj.apply(convert_module_to_f32)
            self.h_skin_coord_fuse.apply(convert_module_to_f32)
        if self.h_skin_unet is not None:
            self.h_skin_unet.apply(convert_module_to_f32)
        self.skin_decoder.convert_to_fp32()
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret
    
    def to_representation_skl(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to skeleton representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of skeleton representations
        """
        ret = []
        for i in range(x.shape[0]):
            skl = self.skeleton_extractor(x[i], training=self.training)
            ret.append(skl)
        return ret

    def forward(self, x: sp.SparseTensor, x_skl: sp.SparseTensor, gt_joints=None, gt_parents=None) -> List[MeshExtractResult]:
        x0 = x
        x_skin = sp.SparseTensor(feats=x0.feats[:, self.latent_channels:], coords=x0.coords.clone())
        x = x0.replace(x0.feats[:, :self.latent_channels])
        if self.normalize_z:
            x_skin = x_skin.replace(F.normalize(x_skin.feats, dim=-1))
            x_skl = x_skl.replace(F.normalize(x_skl.feats, dim=-1))
        
        # Backbone forward
        h, h_skl, h_skin = super().forward(x, x_skl, x_skin)

        # Optional smoothing on h_skin.
        if self.h_skin_unet is not None:
            h_skin = self.h_skin_unet(h_skin)
        
        # Skeleton prediction
        if self.upsample_skl:
            for block_skl in self.upsample_skl_net:
                h_skl = block_skl(h_skl)
        h_skl_middle = h_skl.type(x_skl.dtype)
        h_skl = self.out_layer_skl(h_skl_middle)
        h_skl_skin = self.out_layer_skl_skin(h_skl_middle)
        h_skl = h_skl.replace(torch.cat([h_skl.feats, h_skl_skin.feats], dim=-1))
        skeletons = self.to_representation_skl(h_skl)
        skin_feats_joints_list = self.skeleton_grouping(skeletons, gt_joints=gt_joints, gt_parents=gt_parents)

        # Skin cross with grouped joint features
        if self.skin_cross_from_groupped:
            coords_xyz = h_skin.coords[:, 1:].to(device=h_skin.device, dtype=torch.float32)
            coords_norm = (coords_xyz + 0.5) / self.resolution * 2.0 - 1.0
            coords_pe = self.h_skin_coord_embedder(coords_norm)
            coords_pe = coords_pe.to(device=h_skin.device, dtype=h_skin.feats.dtype)
            coords_pe = self.h_skin_coord_proj(coords_pe)
            h_skin = h_skin.replace(torch.cat([h_skin.feats, coords_pe], dim=-1))
            h_skin = self.h_skin_coord_fuse(h_skin)
            joints_ctx = self._build_processed_joints_context(
                skeletons,
                skin_feats_joints_list,
                device=h_skin.device,
                dtype=h_skin.feats.dtype,
            )
            h_skin = self.skin_cross_groupped_net(h_skin, [joints_ctx])
        for block in self.upsample_skin_net:
            h_skin = block(h_skin)
        h_skin = h_skin.type(x.dtype)
        h_skin = self.out_layer_skin(h_skin)

        # Mesh prediction
        for block in self.upsample:
            h = block(h)
        h_middle = h.type(x.dtype)
        h = self.out_layer(h_middle)
        h = h.replace(torch.cat([h.feats, h_skin.feats], dim=-1))
        meshes = self.to_representation(h)

        self.skinweight_forward(meshes, skeletons, gt_joints=gt_joints, gt_parents=gt_parents)
        return meshes, skeletons

    def _joints_feats_list_to_sparse(
        self,
        joints_feats_list: List[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> sp.SparseTensor:
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        feats_per_batch: List[torch.Tensor] = []
        for joints_feats in joints_feats_list:
            joints_feats = joints_feats.to(device=device, dtype=dtype)
            feats_per_batch.append(joints_feats)
        feats = torch.cat(feats_per_batch, dim=0)
        # Coords are [batch, x, y, z]. We encode token index into x and keep y/z = 0.
        batch_indices: List[torch.Tensor] = []
        x_indices: List[torch.Tensor] = []
        for bi, joints_feats in enumerate(feats_per_batch):
            ji = int(joints_feats.shape[0])
            batch_indices.append(torch.full((ji,), bi, device=device, dtype=torch.int32))
            x_indices.append(torch.arange(ji, device=device, dtype=torch.int32))
        b = torch.cat(batch_indices, dim=0)
        x = torch.cat(x_indices, dim=0)
        yz = torch.zeros((x.shape[0], 2), device=device, dtype=torch.int32)
        coords = torch.cat([b[:, None], x[:, None], yz], dim=1)
        return sp.SparseTensor(feats=feats, coords=coords)

    def _build_processed_joints_context(
        self,
        skeletons: List[Any],
        skin_feats_joints_list: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> sp.SparseTensor:
        processed: List[torch.Tensor] = []
        raw_skin: List[torch.Tensor] = []
        for rep_skl, skin_feats_joints in zip(skeletons, skin_feats_joints_list):
            joints = rep_skl.joints_grouped
            parents = rep_skl.parents_grouped
            if joints is None or parents is None:
                raise ValueError('Expected grouped joints/parents for skin_cross_from_groupped.')
            joints = joints.to(device=device, dtype=dtype)
            parents = parents.to(device=device)
            skin_feats_joints = skin_feats_joints.to(device=device, dtype=dtype)
            raw_skin.append(skin_feats_joints)

            pe = self.joints_pos_embedder(joints).to(device=device, dtype=dtype)

            # Parent skin features (root uses trainable parameter)
            parent_idx = parents.to(torch.long)
            valid = parent_idx >= 0
            root_feat = self.root_parent_feat.to(device=device, dtype=dtype)
            parent_feat_root = root_feat.unsqueeze(0).expand(skin_feats_joints.shape[0], -1)
            parent_feat_gather = skin_feats_joints[parent_idx.clamp(min=0)]
            parent_feat = torch.where(valid.unsqueeze(1), parent_feat_gather, parent_feat_root)

            joint_in = torch.cat([skin_feats_joints, pe, parent_feat], dim=-1)
            joint_h = self.joints_in_proj(joint_in)
            processed.append(joint_h)

        joints_ctx = self._joints_feats_list_to_sparse(processed, device=device, dtype=dtype)
        for blk in self.joints_self_attn:
            joints_ctx = blk(joints_ctx)
        # Skip connection: concatenate original joint skin feats after self-attn.
        joints_skip = self._joints_feats_list_to_sparse(raw_skin, device=device, dtype=dtype)
        joints_ctx = joints_ctx.replace(torch.cat([joints_ctx.feats, joints_skip.feats], dim=-1))
        return joints_ctx
    
    def skeleton_grouping(self, reps_skl, gt_joints=None, gt_parents=None, skin_feats_skl_list=None, return_skin_pred_only=False):
        skin_feats_joints_list = []
        for i, rep_skl in zip(range(len(reps_skl)), reps_skl):
            if gt_joints is not None:
                joints_grouped = gt_joints[i]
                parents_grouped = gt_parents[i]
            elif rep_skl.joints_grouped is None:
                with torch.no_grad():
                    joints_grouped, parents_grouped = self.grouping_func(joints=rep_skl.joints, parents=rep_skl.parents, joints_conf=rep_skl.conf_j, parents_conf=rep_skl.conf_p)
            else:
                joints_grouped = rep_skl.joints_grouped
                parents_grouped = rep_skl.parents_grouped
            
            if not return_skin_pred_only:
                rep_skl.joints_grouped = joints_grouped
                rep_skl.parents_grouped = parents_grouped

            # Calculate NN indices for joints
            positions_skl = rep_skl.positions
            _, joints_nn_idx, _ = knn_points(positions_skl[None], joints_grouped[None].detach(), K=1, norm=2, return_nn=False)
            joints_nn_idx = joints_nn_idx[0, :, 0]
            skin_feats_skl = rep_skl.skin_feats if skin_feats_skl_list is None else skin_feats_skl_list[i]
            
            # Average the predicted joint features
            conf_skin = torch.sigmoid(rep_skl.conf_skin) if rep_skl.conf_skin is not None else torch.ones_like(skin_feats_skl[:, :1])

            skin_feats_joints = torch.zeros([joints_grouped.shape[0], skin_feats_skl.shape[-1]], device=self.device, dtype=skin_feats_skl.dtype)
            skin_feats_square_joints = skin_feats_joints.clone()
            skin_conf_joints = torch.zeros([joints_grouped.shape[0], 1], device=self.device, dtype=skin_feats_skl.dtype)

            skin_feats_joints.scatter_add_(0, joints_nn_idx[:, None].expand(-1, skin_feats_skl.shape[-1]), skin_feats_skl * conf_skin)
            skin_feats_square_joints.scatter_add_(0, joints_nn_idx[:, None].expand(-1, skin_feats_skl.shape[-1]), skin_feats_skl.square() * conf_skin)
            skin_conf_joints.scatter_add_(0, joints_nn_idx[:, None], conf_skin)

            skin_feats_joints = skin_feats_joints / skin_conf_joints.clamp(min=1e-6)
            skin_feats_square_joints = skin_feats_square_joints / skin_conf_joints.clamp(min=1e-6)
            skin_feats_joints_var = skin_feats_square_joints - skin_feats_joints.square()
            skin_feats_joints_var_loss = skin_feats_joints_var.mean()
            
            if not return_skin_pred_only:
                rep_skl.skin_feats_joints_var_loss = skin_feats_joints_var_loss
                rep_skl.skin_feats_joints = skin_feats_joints    
            skin_feats_joints_list.append(skin_feats_joints)
        return skin_feats_joints_list
        
    def skinweight_forward(self, reps, reps_skl, gt_joints=None, gt_parents=None, return_skin_pred_only=False, skin_feats_verts_list=None, skin_feats_skl_list=None, *args, **kwargs):
        if return_skin_pred_only:
            skin_preds = []
        if reps_skl[0].parents_grouped is None or return_skin_pred_only:
            skin_feats_joints_list = self.skeleton_grouping(reps_skl, gt_joints=gt_joints, gt_parents=gt_parents, skin_feats_skl_list=skin_feats_skl_list, return_skin_pred_only=return_skin_pred_only)
        else:
            skin_feats_joints_list = [rep_skl.skin_feats_joints for rep_skl in reps_skl]
        for i, rep, rep_skl in zip(range(len(reps)), reps, reps_skl):
            # Joint skinning features
            skin_feats_joints = skin_feats_joints_list[i]
            # Vertex skinning features
            skin_feats_verts = rep.vertex_skin_feats if skin_feats_verts_list is None else skin_feats_verts_list[i]
            # Predict skin weights
            parents_grouped = rep_skl.parents_grouped
            skin_pred = self.skin_decoder(skin_feats_verts[None], skin_feats_joints[None], parents_grouped[None])
            skin_pred = skin_pred[0]
            if return_skin_pred_only:
                skin_preds.append(skin_pred)
            else:
                reps_skl[i].skin_pred = skin_pred
        if return_skin_pred_only:
            return skin_preds
    

class AniGenElasticSLatMeshDecoder(SparseTransformerElasticMixin, AniGenSLatMeshDecoder):
    """
    Slat VAE Mesh decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
