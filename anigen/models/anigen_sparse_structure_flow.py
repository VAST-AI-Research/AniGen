from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


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


class AniGenSparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        in_channels_skl: int,
        model_channels: int,
        model_channels_skl: int,
        cond_channels: int,
        out_channels: int,
        out_channels_skl: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_pretrain_branch: bool = True,
        freeze_pretrain_branch: bool = True,
        use_lora_ss: bool = False,
        lora_lr_rate_ss: float = 0.1,
        modules_to_freeze: Optional[List[str]] = ["blocks", "input_layer", "out_layer", "pos_emb", "t_embedder"],
        adapter_ss_to_skl: bool = True,
        adapter_skl_to_ss: bool = True,
        predict_x0: bool = False,
        predict_x0_skl: bool = False,
        t_eps: float = 5e-2,
        t_scale: float = 1e3,
        z_is_global: bool = False,
        z_skl_is_global: bool = False,
        global_token_num: int = 1024,
        global_token_num_skl: int = 1024,
        cross_adapter_every: int = 4,
        skl_cross_from_ss: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.in_channels_skl = in_channels_skl
        self.model_channels = model_channels
        self.model_channels_skl = model_channels_skl
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.out_channels_skl = out_channels_skl
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.use_pretrain_branch = use_pretrain_branch
        self.freeze_pretrain_branch = freeze_pretrain_branch or use_lora_ss
        self.use_lora_ss = use_lora_ss
        self.modules_to_freeze = modules_to_freeze
        self.adapter_ss_to_skl = adapter_ss_to_skl
        self.adapter_skl_to_ss = adapter_skl_to_ss
        self.predict_x0 = predict_x0
        self.predict_x0_skl = predict_x0_skl
        self.t_eps = t_eps
        self.t_scale = t_scale
        self.z_is_global = z_is_global
        self.z_skl_is_global = z_skl_is_global
        self.global_token_num = global_token_num
        self.global_token_num_skl = global_token_num_skl
        self.cross_adapter_every = int(cross_adapter_every)
        self.skl_cross_from_ss = skl_cross_from_ss

        self.t_embedder = TimestepEmbedder(model_channels)
        self.t_embedder_skl = TimestepEmbedder(model_channels_skl)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
            self.adaLN_modulation_skl = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels_skl, 6 * model_channels_skl, bias=True)
            )

        if pe_mode == "ape":
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            if self.z_is_global:
                pos_embedder = AbsolutePositionEmbedder(model_channels, 1)
                pos_emb = pos_embedder(torch.arange(self.global_token_num, device=self.device)[:, None])
            else:
                pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
                pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)
            if self.z_skl_is_global:
                pos_embedder_skl = AbsolutePositionEmbedder(model_channels_skl, 1)
                pos_emb_skl = pos_embedder_skl(torch.arange(self.global_token_num_skl, device=self.device)[:, None])
            else:
                pos_embedder_skl = AbsolutePositionEmbedder(model_channels_skl, 3)
                pos_emb_skl = pos_embedder_skl(coords)
            self.register_buffer("pos_emb_skl", pos_emb_skl)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.input_layer_skl = nn.Linear(in_channels_skl * patch_size**3, model_channels_skl)
        
        shallow = max(1, num_blocks // 3)
        middle = max(1, num_blocks // 3 * 2)
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                use_lora_self=self.use_lora_ss and idx >= middle,
                lora_rank_self=8,
                use_lora_cross=self.use_lora_ss,
                lora_rank_cross=8+(idx // shallow)*8,
                lora_lr_rate=lora_lr_rate_ss,
            )
            for idx in range(num_blocks)
        ])
        self.blocks_skl = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels_skl,
                cond_channels if not self.skl_cross_from_ss else model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                use_context_norm=self.skl_cross_from_ss,
            )
            for _ in range(num_blocks)
        ])

        # When using global tokens, ss and skl token counts may differ, so we use cross-attention
        # for information exchange at a configurable frequency.
        self.use_cross_adapter = (self.z_is_global or self.z_skl_is_global) and (
            self.adapter_ss_to_skl or self.adapter_skl_to_ss
        )

        if self.adapter_ss_to_skl and not self.use_cross_adapter:
            self.adapter_ss_to_skl_layers = nn.ModuleList([
                nn.Linear(model_channels, model_channels_skl) for _ in range(num_blocks)
            ])
        if self.adapter_skl_to_ss and not self.use_cross_adapter:
            self.adapter_skl_to_ss_layers = nn.ModuleList([
                nn.Linear(model_channels_skl, model_channels) for _ in range(num_blocks)
            ])
        
        self.cross_adapter_every = max(1, self.cross_adapter_every)
        self.cross_block_indices: List[int] = [
            idx for idx in range(num_blocks) if (idx + 1) % self.cross_adapter_every == 0
        ]
        if self.use_cross_adapter and len(self.cross_block_indices) == 0 and num_blocks > 0:
            self.cross_block_indices = [num_blocks - 1]
        if self.use_cross_adapter and len(self.cross_block_indices) > 0:
            if self.adapter_ss_to_skl:
                self.cross_blocks_ss_to_skl = nn.ModuleList([
                    ModulatedTransformerCrossBlock(
                        model_channels_skl,
                        model_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        attn_mode='full',
                        use_checkpoint=self.use_checkpoint,
                        use_rope=(pe_mode == "rope"),
                        share_mod=share_mod,
                        qk_rms_norm=self.qk_rms_norm,
                        qk_rms_norm_cross=self.qk_rms_norm_cross,
                    )
                    for _ in self.cross_block_indices
                ])
                self.cross_blocks_ss_to_skl_out = nn.ModuleList([
                    nn.Linear(model_channels_skl, model_channels_skl, bias=True)
                    for _ in self.cross_block_indices
                ])
            if self.adapter_skl_to_ss:
                self.cross_blocks_skl_to_ss = nn.ModuleList([
                    ModulatedTransformerCrossBlock(
                        model_channels,
                        model_channels_skl,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        attn_mode='full',
                        use_checkpoint=self.use_checkpoint,
                        use_rope=(pe_mode == "rope"),
                        share_mod=share_mod,
                        qk_rms_norm=self.qk_rms_norm,
                        qk_rms_norm_cross=self.qk_rms_norm_cross,
                    )
                    for _ in self.cross_block_indices
                ])
                self.cross_blocks_skl_to_ss_out = nn.ModuleList([
                    nn.Linear(model_channels, model_channels, bias=True)
                    for _ in self.cross_block_indices
                ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)
        self.out_layer_skl = nn.Linear(model_channels_skl, out_channels_skl * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
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
        self.blocks.apply(convert_module_to_f16)
        self.blocks_skl.apply(convert_module_to_f16)
        if hasattr(self, "adapter_ss_to_skl_layers"):
            self.adapter_ss_to_skl_layers.apply(convert_module_to_f16)
        if hasattr(self, "adapter_skl_to_ss_layers"):
            self.adapter_skl_to_ss_layers.apply(convert_module_to_f16)
        if getattr(self, "use_cross_adapter", False):
            if hasattr(self, "cross_blocks_ss_to_skl"):
                self.cross_blocks_ss_to_skl.apply(convert_module_to_f16)
                self.cross_blocks_ss_to_skl_out.apply(convert_module_to_f16)
            if hasattr(self, "cross_blocks_skl_to_ss"):
                self.cross_blocks_skl_to_ss.apply(convert_module_to_f16)
                self.cross_blocks_skl_to_ss_out.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        self.blocks_skl.apply(convert_module_to_f32)
        if hasattr(self, "adapter_ss_to_skl_layers"):
            self.adapter_ss_to_skl_layers.apply(convert_module_to_f32)
        if hasattr(self, "adapter_skl_to_ss_layers"):
            self.adapter_skl_to_ss_layers.apply(convert_module_to_f32)
        if getattr(self, "use_cross_adapter", False):
            if hasattr(self, "cross_blocks_ss_to_skl"):
                self.cross_blocks_ss_to_skl.apply(convert_module_to_f32)
                self.cross_blocks_ss_to_skl_out.apply(convert_module_to_f32)
            if hasattr(self, "cross_blocks_skl_to_ss"):
                self.cross_blocks_skl_to_ss.apply(convert_module_to_f32)
                self.cross_blocks_skl_to_ss_out.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder_skl.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_skl.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.adaLN_modulation_skl[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation_skl[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.blocks_skl:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
        nn.init.constant_(self.out_layer_skl.weight, 0)
        nn.init.constant_(self.out_layer_skl.bias, 0)

        # Zero-out adapter layers if exist
        if hasattr(self, "adapter_ss_to_skl_layers"):
            for layer in self.adapter_ss_to_skl_layers:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
        if hasattr(self, "adapter_skl_to_ss_layers"):
            for layer in self.adapter_skl_to_ss_layers:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)

        # Zero-out cross adapter output projections (so we can safely finetune from pretrained ckpt)
        if getattr(self, "use_cross_adapter", False):
            if hasattr(self, "cross_blocks_ss_to_skl_out"):
                for layer in self.cross_blocks_ss_to_skl_out:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if hasattr(self, "cross_blocks_skl_to_ss_out"):
                for layer in self.cross_blocks_skl_to_ss_out:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor, x_skl: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.z_is_global:
            assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                    f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"        
        if not self.z_skl_is_global:
            assert [*x_skl.shape] == [x_skl.shape[0], self.in_channels_skl, *[self.resolution] * 3], \
                    f"Input shape mismatch, got {x_skl.shape}, expected {[x_skl.shape[0], self.in_channels_skl, *[self.resolution] * 3]}"
        
        if self.predict_x0:
            xt = x.clone()
        if self.predict_x0_skl:
            xt_skl = x_skl.clone()

        if not self.z_is_global:
            h = patchify(x, self.patch_size)
            h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        else:
            h = x
        if not self.z_skl_is_global:
            h_skl = patchify(x_skl, self.patch_size)
            h_skl = h_skl.view(*h_skl.shape[:2], -1).permute(0, 2, 1).contiguous()
        else:
            h_skl = x_skl

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        h_skl = self.input_layer_skl(h_skl)
        h_skl = h_skl + self.pos_emb_skl[None]

        t_emb = self.t_embedder(t)
        t_emb_skl = self.t_embedder_skl(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
            t_emb_skl = self.adaLN_modulation_skl(t_emb_skl)
        t_emb = t_emb.type(self.dtype)
        t_emb_skl = t_emb_skl.type(self.dtype)

        h = h.type(self.dtype)
        h_skl = h_skl.type(self.dtype)
        cond = cond.type(self.dtype)

        cross_pos_to_idx = None
        if self.use_cross_adapter and len(self.cross_block_indices) > 0:
            cross_pos_to_idx = {bidx: cidx for cidx, bidx in enumerate(self.cross_block_indices)}

        for idx, block, block_skl in zip(range(len(self.blocks)), self.blocks, self.blocks_skl):
            f = block(h, t_emb, cond)
            f_skl = block_skl(h_skl, t_emb_skl, h if self.skl_cross_from_ss else cond)

            if self.use_cross_adapter and cross_pos_to_idx is not None and idx in cross_pos_to_idx:
                cidx = cross_pos_to_idx[idx]
                if self.adapter_ss_to_skl:
                    out_skl = self.cross_blocks_ss_to_skl[cidx](f_skl, t_emb_skl, f)
                    h_skl = f_skl + self.cross_blocks_ss_to_skl_out[cidx](out_skl - f_skl)
                else:
                    h_skl = f_skl

                if self.adapter_skl_to_ss:
                    out = self.cross_blocks_skl_to_ss[cidx](f, t_emb, f_skl)
                    h = f + self.cross_blocks_skl_to_ss_out[cidx](out - f)
                else:
                    h = f
            else:
                # Non-global (or no cross block at this idx): keep previous behavior.
                if self.adapter_ss_to_skl and (not self.use_cross_adapter):
                    h_skl = f_skl + self.adapter_ss_to_skl_layers[idx](f)
                else:
                    h_skl = f_skl

                if self.adapter_skl_to_ss and (not self.use_cross_adapter):
                    h = f + self.adapter_skl_to_ss_layers[idx](f_skl)
                else:
                    h = f
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h_skl = h_skl.type(x_skl.dtype)
        h_skl = F.layer_norm(h_skl, h_skl.shape[-1:])
        h_skl = self.out_layer_skl(h_skl)

        if not self.z_is_global:
            h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
            h = unpatchify(h, self.patch_size).contiguous()

        if not self.z_skl_is_global:
            h_skl = h_skl.permute(0, 2, 1).view(h_skl.shape[0], h_skl.shape[2], *[self.resolution // self.patch_size] * 3)
            h_skl = unpatchify(h_skl, self.patch_size).contiguous()

        if self.predict_x0:
            t_normalized = t / self.t_scale
            factor = (1 / t_normalized.clamp_min(self.t_eps)).reshape([t.shape[0], *([1] * (x.dim() - 1))])
            h = (xt - h) * factor
        if self.predict_x0_skl:
            t_normalized = t / self.t_scale
            factor = (1 / t_normalized.clamp_min(self.t_eps)).reshape([t.shape[0], *([1] * (x_skl.dim() - 1))])
            h_skl = (xt_skl - h_skl) * factor

        return h, h_skl
