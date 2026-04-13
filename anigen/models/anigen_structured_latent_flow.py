from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from anigen.modules.transformer import blocks
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_elastic_mixin import SparseTransformerElasticMixin


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h


class AniGenSLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        in_channels_vert_skin: int,
        in_channels_skl: int,
        model_channels: int,
        model_channels_vert_skin: int,
        model_channels_skl: int,
        cond_channels: int,
        out_channels: int,
        out_channels_vert_skin: int,
        out_channels_skl: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        num_heads_vert_skin: Optional[int] = None,
        num_head_channels_vert_skin: Optional[int] = 64,
        num_heads_skl: Optional[int] = None,
        num_head_channels_skl: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        num_io_res_blocks_vert_skin: int = 2,
        num_io_res_blocks_skl: int = 2,
        io_block_channels: List[int] = None,
        io_block_channels_vert_skin: List[int] = None,
        io_block_channels_skl: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_pretrain_branch: bool = True,
        freeze_pretrain_branch: bool = True,
        modules_to_freeze: Optional[List[str]] = ['blocks', 'input_blocks','input_layer', 'out_blocks', 'out_layer', 't_embedder'],
        predict_x0: bool = False,
        t_eps: float = 5e-2,
        t_scale: float = 1e3,
        use_joint_num_cond: bool = False,
        joint_num_max: int = 60,
        joint_num_fourier_bands: int = 6,
    ):
        super().__init__()
        
        self.pretrain_class_name = ["AniGenSlatFlowImage"]

        self.resolution = resolution
        self.in_channels = in_channels
        self.in_channels_vert_skin = in_channels_vert_skin
        self.in_channels_skl = in_channels_skl
        self.model_channels = model_channels
        self.model_channels_vert_skin = model_channels_vert_skin
        self.model_channels_skl = model_channels_skl
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.out_channels_vert_skin = out_channels_vert_skin
        self.out_channels_skl = out_channels_skl
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.num_heads_vert_skin = num_heads_vert_skin or model_channels_vert_skin // num_head_channels_vert_skin
        self.num_heads_skl = num_heads_skl or model_channels_skl // num_head_channels_skl
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.num_io_res_blocks_vert_skin = num_io_res_blocks_vert_skin
        self.num_io_res_blocks_skl = num_io_res_blocks_skl
        self.io_block_channels = io_block_channels
        self.io_block_channels_vert_skin = io_block_channels_vert_skin
        self.io_block_channels_skl = io_block_channels_skl
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.predict_x0 = predict_x0
        self.t_eps = t_eps
        self.t_scale = t_scale
        self.use_joint_num_cond = use_joint_num_cond
        self.joint_num_max = joint_num_max
        self.joint_num_fourier_bands = joint_num_fourier_bands

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        self.t_embedder_vert_skin = TimestepEmbedder(model_channels_vert_skin)
        self.t_embedder_skl = TimestepEmbedder(model_channels_skl)

        if self.use_joint_num_cond:
            # Joint-number conditioning (applied to skin + skeleton branches).
            # If joints_num is missing/<=0, use learnable unconditional embeddings.
            self.joint_num_embedder_vert_skin = nn.Sequential(
                nn.Linear(2 * joint_num_fourier_bands, model_channels_vert_skin, bias=True),
                nn.SiLU(),
                nn.Linear(model_channels_vert_skin, model_channels_vert_skin, bias=True),
            )
            self.joint_num_embedder_skl = nn.Sequential(
                nn.Linear(2 * joint_num_fourier_bands, model_channels_skl, bias=True),
                nn.SiLU(),
                nn.Linear(model_channels_skl, model_channels_skl, bias=True),
            )
            self.joint_num_uncond_vert_skin = nn.Parameter(torch.zeros(model_channels_vert_skin))
            self.joint_num_uncond_skl = nn.Parameter(torch.zeros(model_channels_skl))
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
            self.adaLN_modulation_vert_skin = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels_vert_skin, 6 * model_channels_vert_skin, bias=True)
            )
            self.adaLN_modulation_skl = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels_skl, 6 * model_channels_skl, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
            self.pos_embedder_vert_skin = AbsolutePositionEmbedder(model_channels_vert_skin)
            self.pos_embedder_skl = AbsolutePositionEmbedder(model_channels_skl)

        # Causuality in conditioning:
        # Geometry <- Conditioned Image (Cross Attention)
        # Skinning <- Geometry (Adapter Layer) + Skeleton (Cross Attention)
        # Skeleton <- Skinning (Cross Attention)
        causial_cond_channels_dict = {'': cond_channels, '_vert_skin': self.model_channels_skl, '_skl': self.model_channels_vert_skin}

        for postfix in ['', '_vert_skin', '_skl']:
            # Input blocks
            setattr(self, f'input_layer{postfix}', sp.SparseLinear(
                getattr(self, f'in_channels{postfix}'),
                getattr(self, f'model_channels{postfix}') if getattr(self, f'io_block_channels{postfix}') is None else getattr(self, f'io_block_channels{postfix}')[0]
            ))

            setattr(self, f'input_blocks{postfix}', nn.ModuleList([]))
            io_block_channels = getattr(self, f'io_block_channels{postfix}')
            model_channels = getattr(self, f'model_channels{postfix}')
            num_io_res_blocks = getattr(self, f'num_io_res_blocks{postfix}')
            if io_block_channels is not None:
                for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                    getattr(self, f'input_blocks{postfix}').extend([
                        SparseResBlock3d(
                            chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
                    getattr(self, f'input_blocks{postfix}').append(
                        SparseResBlock3d(
                            chs,
                            model_channels,
                            out_channels=next_chs,
                            downsample=True,
                        )
                    )
            
            # Transformer blocks
            cond_channels_block = causial_cond_channels_dict[postfix]
            setattr(self, f'blocks{postfix}', nn.ModuleList([
                ModulatedSparseTransformerCrossBlock(
                    getattr(self, f'model_channels{postfix}'),
                    cond_channels_block,
                    num_heads=getattr(self, f'num_heads{postfix}'),
                    mlp_ratio=self.mlp_ratio,
                    attn_mode='full',
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=self.share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                    norm_for_context=True,
                )
                for _ in range(num_blocks)
            ]))

            # Output blocks
            setattr(self, f'out_blocks{postfix}', nn.ModuleList([]))
            if io_block_channels is not None:
                for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                    getattr(self, f'out_blocks{postfix}').append(
                        SparseResBlock3d(
                            prev_chs * 2 if self.use_skip_connection else prev_chs,
                            model_channels,
                            out_channels=chs,
                            upsample=True,
                        )
                    )
                    getattr(self, f'out_blocks{postfix}').extend([
                        SparseResBlock3d(
                            chs * 2 if self.use_skip_connection else chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks-1)
                    ])
            setattr(self, f'out_layer{postfix}', sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], getattr(self, f'out_channels{postfix}')))

        self.adapter_geo_to_skin = nn.ModuleList([
            sp.SparseLinear(self.model_channels, self.model_channels_vert_skin) for _ in range(num_blocks)
        ])

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
        self.use_pretrain_branch = use_pretrain_branch
        self.freeze_pretrain_branch = freeze_pretrain_branch
        # self.is_geometry_branch_frozen = self.use_pretrain_branch and self.freeze_pretrain_branch and all([module in modules_to_freeze for module in ['blocks', 'input_blocks','input_layer', 'out_blocks', 'out_layer', 't_embedder']])
        
        if self.use_pretrain_branch and self.freeze_pretrain_branch:
            for module in modules_to_freeze:
                if hasattr(self, module):
                    mod = getattr(self, module)
                    if isinstance(mod, nn.ModuleList):
                        for m in mod:
                            for param in m.parameters():
                                param.requires_grad = False
                    else:
                        for param in mod.parameters():
                            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        for postfix in ['', '_vert_skin', '_skl']:
            getattr(self, f'input_blocks{postfix}').apply(convert_module_to_f16)
            getattr(self, f'blocks{postfix}').apply(convert_module_to_f16)
            getattr(self, f'out_blocks{postfix}').apply(convert_module_to_f16)
        self.adapter_geo_to_skin.apply(convert_module_to_f16)
        if self.use_joint_num_cond:
            self.joint_num_embedder_vert_skin.apply(convert_module_to_f16)
            self.joint_num_embedder_skl.apply(convert_module_to_f16)
            self.joint_num_uncond_vert_skin.data = self.joint_num_uncond_vert_skin.data.half()
            self.joint_num_uncond_skl.data = self.joint_num_uncond_skl.data.half()

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        for postfix in ['', '_vert_skin', '_skl']:
            getattr(self, f'input_blocks{postfix}').apply(convert_module_to_f32)
            getattr(self, f'blocks{postfix}').apply(convert_module_to_f32)
            getattr(self, f'out_blocks{postfix}').apply(convert_module_to_f32)
        self.adapter_geo_to_skin.apply(convert_module_to_f32)
        if self.use_joint_num_cond:
            self.joint_num_embedder_vert_skin.apply(convert_module_to_f32)
            self.joint_num_embedder_skl.apply(convert_module_to_f32)
            self.joint_num_uncond_vert_skin.data = self.joint_num_uncond_vert_skin.data.float()
            self.joint_num_uncond_skl.data = self.joint_num_uncond_skl.data.float()

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for postfix in ['', '_vert_skin', '_skl']:
            nn.init.normal_(getattr(self, f't_embedder{postfix}').mlp[0].weight, std=0.02)
            nn.init.normal_(getattr(self, f't_embedder{postfix}').mlp[2].weight, std=0.02)
            if self.share_mod:
                nn.init.constant_(getattr(self, f'adaLN_modulation{postfix}')[-1].weight, 0)
                nn.init.constant_(getattr(self, f'adaLN_modulation{postfix}')[-1].bias, 0)
            else:
                for block in getattr(self, f'blocks{postfix}'):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(getattr(self, f'out_layer{postfix}').weight, 0)
            nn.init.constant_(getattr(self, f'out_layer{postfix}').bias, 0)
        
        for layer in self.adapter_geo_to_skin:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

        if self.use_joint_num_cond:
            # Joint-number conditioning layers
            for emb in [self.joint_num_embedder_vert_skin, self.joint_num_embedder_skl]:
                for m in emb.modules():
                    if isinstance(m, nn.Linear):
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def _fourier_encode_joint_num(self, joints_num: torch.Tensor) -> torch.Tensor:
        """Fourier features for joints_num in [0, joint_num_max]."""
        # Keep dtype consistent with model (e.g., fp16) to avoid Linear dtype mismatch.
        dtype = getattr(self, 'dtype', torch.float32)
        x = (joints_num.to(dtype=dtype) / float(self.joint_num_max)).clamp(0.0, 1.0)
        x = x[:, None]
        freqs = (2.0 ** torch.arange(self.joint_num_fourier_bands, device=x.device, dtype=x.dtype)) * math.pi
        angles = x * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _get_joint_num_emb(self, joints_num: Optional[torch.Tensor], batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (emb_vert_skin, emb_skl), shape [B, C_*]."""
        if joints_num is None:
            joints_num = torch.zeros(batch_size, device=device)
        elif not torch.is_tensor(joints_num):
            joints_num = torch.tensor(joints_num, device=device)
        joints_num = joints_num.to(device=device)
        if joints_num.dim() == 0:
            joints_num = joints_num[None].expand(batch_size)
        joints_num = joints_num.reshape(batch_size)

        mask_dtype = getattr(self, 'dtype', torch.float32)
        uncond_mask = (joints_num <= 0).to(dtype=mask_dtype, device=device)[:, None]
        joints_num = joints_num.clamp(min=0, max=self.joint_num_max)

        fourier = self._fourier_encode_joint_num(joints_num)
        emb_vs_cond = self.joint_num_embedder_vert_skin(fourier)
        emb_skl_cond = self.joint_num_embedder_skl(fourier)

        emb_vs_uncond = self.joint_num_uncond_vert_skin[None].expand(batch_size, -1)
        emb_skl_uncond = self.joint_num_uncond_skl[None].expand(batch_size, -1)

        # Blend: uncond_mask==1 -> unconditional, uncond_mask==0 -> conditional.
        emb_vs = emb_vs_cond * (1.0 - uncond_mask) + emb_vs_uncond * uncond_mask
        emb_skl = emb_skl_cond * (1.0 - uncond_mask) + emb_skl_uncond * uncond_mask
        return emb_vs, emb_skl

    def forward_stage(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        postfix,
        stage,
        cond_emb: Optional[torch.Tensor] = None,
        t_emb=None,
        skips=None,
        original_dtype=None,
    ) -> sp.SparseTensor:
        input_layer  = getattr(self, f'input_layer{postfix}')
        t_embedder   = getattr(self, f't_embedder{postfix}')
        input_blocks = getattr(self, f'input_blocks{postfix}')
        pos_embedder = getattr(self, f'pos_embedder{postfix}')
        out_blocks   = getattr(self, f'out_blocks{postfix}')
        out_layer    = getattr(self, f'out_layer{postfix}')
        adaLN_modulation = getattr(self, f'adaLN_modulation{postfix}') if self.share_mod else None

        if stage == 'in':
            h = input_layer(x).type(self.dtype)
            t_emb = t_embedder(t)
            if cond_emb is not None:
                t_emb = t_emb + cond_emb
            t_emb = t_emb.type(self.dtype)
            t_mod = adaLN_modulation(t_emb).type(self.dtype) if self.share_mod else t_emb
            skips = []
            # pack with input blocks
            for block in input_blocks:
                h = block(h, t_emb)
                skips.append(h.feats)
            if self.pe_mode == "ape":
                h = h + pos_embedder(h.coords[:, 1:]).type(self.dtype)
            return h, t_emb, t_mod, skips
        elif stage == 'out':
            h = x
            # unpack with output blocks
            for block, skip in zip(out_blocks, reversed(skips)):
                if self.use_skip_connection:
                    h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
                else:
                    h = block(h, t_emb)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = out_layer(h.type(original_dtype))
            return h
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def forward(self, x: sp.SparseTensor, x_skl: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, joints_num: Optional[torch.Tensor] = None, **kwargs) -> sp.SparseTensor:
        cond = cond.type(self.dtype)
        feats, feats_vert_skin = x.feats[:, :self.in_channels], x.feats[:, self.in_channels:]
        x, x_vert_skin = x.replace(feats), x.replace(feats_vert_skin)
        if self.predict_x0:
            xt_feats_skin, xt_feats_skl = feats_vert_skin.clone(), x_skl.feats.clone()

        joint_emb_vs, joint_emb_skl = None, None
        if self.use_joint_num_cond:
            # joint-number conditioning for skin + skeleton
            joint_emb_vs, joint_emb_skl = self._get_joint_num_emb(joints_num, x.shape[0], x.device)
            joint_emb_vs = joint_emb_vs.type(self.dtype)
            joint_emb_skl = joint_emb_skl.type(self.dtype)

        in_dicts = {'': x, '_vert_skin': x_vert_skin, '_skl': x_skl}
        cond_emb_dicts = {'': None, '_vert_skin': joint_emb_vs, '_skl': joint_emb_skl}
        postfix_keys = list(in_dicts.keys())
        for postfix in postfix_keys:
            cond_emb = cond_emb_dicts[postfix]
            in_dicts[postfix], in_dicts[f't_emb{postfix}'], in_dicts[f't_mod{postfix}'], in_dicts[f'skips{postfix}'] = self.forward_stage(in_dicts[postfix], t, postfix, stage='in', cond_emb=cond_emb)
        for block, block_skin, block_skl, adapter in zip(self.blocks, self.blocks_vert_skin, self.blocks_skl, self.adapter_geo_to_skin):
            h, h_skin, h_skl = in_dicts[''], in_dicts['_vert_skin'], in_dicts['_skl']
            f = block(h, in_dicts['t_mod'], cond)
            f_skin = block_skin(h_skin, in_dicts['t_mod_vert_skin'], h_skl) + adapter(h)
            f_skl = block_skl(h_skl, in_dicts['t_mod_skl'], h_skin)
            in_dicts[''], in_dicts['_vert_skin'], in_dicts['_skl'] = f, f_skin, f_skl
        for postfix in postfix_keys:
            in_dicts[postfix] = self.forward_stage(
                in_dicts[postfix],
                t,
                postfix,
                stage='out',
                t_emb=in_dicts[f't_emb{postfix}'],
                skips=in_dicts[f'skips{postfix}'],
                original_dtype=x.dtype,
            )
        if self.predict_x0:
            t_normalized = t / self.t_scale
            factor = (1 / t_normalized.clamp_min(self.t_eps))[:, None]
            in_dicts['_vert_skin'] = in_dicts['_vert_skin'].replace((in_dicts['_vert_skin'].feats - xt_feats_skin) * factor[in_dicts['_vert_skin'].coords[:, 0]])
            in_dicts['_skl'] = in_dicts['_skl'].replace((in_dicts['_skl'].feats - xt_feats_skl) * factor[in_dicts['_skl'].coords[:, 0]])
        x_out = x.replace(torch.cat([in_dicts[''].feats, in_dicts['_vert_skin'].feats], dim=1))
        x_skl_out = x_skl.replace(in_dicts['_skl'].feats)
        return x_out, x_skl_out

class AniGenElasticSLatFlowModel(SparseTransformerElasticMixin, AniGenSLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass
