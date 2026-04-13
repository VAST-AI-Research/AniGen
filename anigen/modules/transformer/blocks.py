from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, out_channels: Optional[int] = None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        out_channels: Optional[int] = None,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.mlp = FeedForwardNet(
            self.channels,
            out_channels=self.out_channels,
            mlp_ratio=mlp_ratio,
        )
        if self.out_channels != self.channels:
            self.res_mlp = FeedForwardNet(
                self.channels,
                out_channels=self.out_channels,
                mlp_ratio=1.0,
            )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        if self.out_channels != self.channels:
            x = self.res_mlp(x)
        x = x + h
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SkinTransformerCrossBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        out_channels: Optional[int] = None,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.to_v = nn.Linear(channels, channels, bias=qkv_bias)
        self.channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.mlp = FeedForwardNet(
            self.channels,
            out_channels=self.out_channels,
            mlp_ratio=mlp_ratio,
        )
        self.joint_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode="full",
            qkv_bias=qkv_bias,
        )
        self.joint_mlp = FeedForwardNet(
            self.channels,
            out_channels=self.out_channels,
            mlp_ratio=mlp_ratio,
        )
        if self.out_channels != self.channels:
            self.res_mlp = FeedForwardNet(
                self.channels,
                out_channels=self.out_channels,
                mlp_ratio=1.0,
            )
            self.res_joint_mlp = FeedForwardNet(
                self.channels,
                out_channels=self.out_channels,
                mlp_ratio=1.0,
            )

    def _forward(self, x: torch.Tensor, j: torch.Tensor, skin: torch.Tensor) -> torch.Tensor:
        v = self.to_v(self.norm1(j))
        h = skin @ v
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        if self.out_channels != self.channels:
            x = self.res_mlp(x)
        x = x + h

        h_j = self.norm3(j)
        h_j = self.joint_attn(h_j)
        h_j = j + h_j
        h_j = self.joint_mlp(h_j)
        if self.out_channels != self.channels:
            j = self.res_joint_mlp(j)
        j = j + h_j
        return x, j


class TransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        out_channels: Optional[int] = None,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        x_is_query: bool = False,
        no_self: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6) if not no_self else nn.Identity()
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        if no_self:
            self.self_attn = lambda x: 0
        else:
            self.self_attn = MultiHeadAttention(
                channels,
                num_heads=num_heads,
                type="self",
                attn_mode=attn_mode,
                window_size=window_size,
                shift_window=shift_window,
                qkv_bias=qkv_bias,
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
            )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            x_is_query=x_is_query,
        )
        self.channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.mlp = FeedForwardNet(
            channels,
            out_channels=self.out_channels,
            mlp_ratio=mlp_ratio,
        )
        if self.out_channels != self.channels:
            self.res_mlp = FeedForwardNet(
                self.channels,
                out_channels=self.out_channels,
                mlp_ratio=1.0,
            )

    def _forward(self, x: torch.Tensor, context: torch.Tensor):
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        if self.out_channels != self.channels:
            x = self.res_mlp(x)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)
        