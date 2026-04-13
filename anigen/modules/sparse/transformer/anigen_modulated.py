from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class AniGenModulatedSparseTransformerCrossBlock(nn.Module):
    """
    AniGen Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        channels_skl: int,
        ctx_channels: int,
        num_heads: int,
        num_heads_skl: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,

    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm1_skl = LayerNorm32(channels_skl, elementwise_affine=False, eps=1e-6)
        self.norm2_skl = LayerNorm32(channels_skl, elementwise_affine=True, eps=1e-6)
        self.norm3_skl = LayerNorm32(channels_skl, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=channels_skl,
            num_heads=num_heads,
            type="cross",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.attn_skl = SparseMultiHeadAttention(
            channels_skl,
            ctx_channels=channels,
            num_heads=num_heads_skl,
            type="cross",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.context_cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.context_cross_attn_skl = SparseMultiHeadAttention(
            channels_skl,
            ctx_channels=ctx_channels,
            num_heads=num_heads_skl,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        self.mlp_skl = SparseFeedForwardNet(
            channels_skl,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
            self.adaLN_modulation_skl = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels_skl, 6 * channels_skl, bias=True)
            )

    def _forward(self, x: SparseTensor, x_skl: SparseTensor, mod: torch.Tensor, mod_skl: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
            shift_msa_skl, scale_msa_skl, gate_msa_skl, shift_mlp_skl, scale_mlp_skl, gate_mlp_skl = mod_skl.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            shift_msa_skl, scale_msa_skl, gate_msa_skl, shift_mlp_skl, scale_mlp_skl, gate_mlp_skl = self.adaLN_modulation_skl(mod_skl).chunk(6, dim=1)
        # Input Norm
        h = x.replace(self.norm1(x.feats))
        h_skl = x_skl.replace(self.norm1_skl(x_skl.feats))
        # AdaLN (By Time Step)
        h = h * (1 + scale_msa) + shift_msa
        h_skl = h_skl * (1 + scale_msa_skl) + shift_msa_skl
        # Self Attn (Cross shape and skeleton)
        h = self.attn(h, h_skl)
        h_skl = self.attn_skl(h_skl, h)
        # Gated Residual (By Time Step)
        h = h * gate_msa
        h_skl = h_skl * gate_msa_skl
        x = x + h
        x_skl = x_skl + h_skl
        # Context Cross Attention
        h = x.replace(self.norm2(x.feats))
        h_skl = x_skl.replace(self.norm2_skl(x_skl.feats))
        h = self.context_cross_attn(h, context)
        h_skl = self.context_cross_attn_skl(h_skl, context)
        x = x + h
        x_skl = x_skl + h_skl
        # Re-Centered
        h = x.replace(self.norm3(x.feats))
        h_skl = x_skl.replace(self.norm3_skl(x_skl.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h_skl = h_skl * (1 + scale_mlp_skl) + shift_mlp_skl
        # Output MLP
        h = self.mlp(h)
        h_skl = self.mlp_skl(h_skl)
        # Gated Residual (By Time Step)
        h = h * gate_mlp
        h_skl = h_skl * gate_mlp_skl
        x = x + h
        x_skl = x_skl + h_skl
        return x, x_skl

    def forward(self, x: SparseTensor, x_skl: SparseTensor, mod: torch.Tensor, mod_skl: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, x_skl, mod, mod_skl, context, use_reentrant=False)
        else:
            return self._forward(x, x_skl, mod, mod_skl, context)
