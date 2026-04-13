from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import FeedForwardNet, TransformerBlock, TransformerCrossBlock, AbsolutePositionEmbedder


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        

class AniGenSparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).
    
    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        in_channels: int,
        in_channels_skl: int,
        latent_channels: int,
        latent_channels_skl: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        encode_global: bool = False,
        global_token_num: int = 1024,
        encode_global_skl: bool = True,
        global_token_num_skl: int = 1024,
        use_pretrain_branch: bool = True,
        freeze_pretrain_branch: bool = True,
        modules_to_freeze: Optional[List[str]] = ["input_layer", "blocks", "middle_block", "out_layer"],
        latent_denoising: bool = False,
        latent_denoising_skl: bool = True,
        normalize_z: bool = False,
        normalize_z_skl: bool = True,
        normalize_scale: float = 1.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_skl = in_channels_skl
        self.latent_channels = latent_channels
        self.latent_channels_skl = latent_channels_skl
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.encode_global = encode_global
        self.global_token_num = global_token_num
        self.encode_global_skl = encode_global_skl
        self.global_token_num_skl = global_token_num_skl
        self.use_pretrain_branch = use_pretrain_branch
        self.freeze_pretrain_branch = freeze_pretrain_branch
        self.latent_denoising = latent_denoising
        self.latent_denoising_skl = latent_denoising_skl
        self.normalize_latent = normalize_z and latent_denoising
        self.normalize_latent_skl = normalize_z_skl and latent_denoising_skl
        self.normalize_scale = normalize_scale

        self.input_layer = nn.Conv3d(self.in_channels, channels[0], 3, padding=1)
        self.input_layer_skl = nn.Conv3d(self.in_channels_skl, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        self.blocks_skl = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            self.blocks_skl.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
                self.blocks_skl.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])
        self.middle_block_skl = nn.Sequential(*[
            ResBlock3d(channels[-1] if _ == 0 else channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        if self.encode_global:
            # Initial Tokens and PE
            self.init_tokens_ss = nn.Parameter(torch.zeros(1, global_token_num, channels[-1]))
            pos_embedder = AbsolutePositionEmbedder(channels[-1], 1)
            coords = torch.arange(global_token_num, device=self.device).reshape(-1, 1)
            tokens_pos_emb = pos_embedder(coords)
            self.register_buffer('tokens_pos_emb_ss', tokens_pos_emb)
            # Grids PE
            upsample_factor = 2 ** (len(channels) - 1)
            self.base_size_ss = 64 // upsample_factor
            pos_embedder = AbsolutePositionEmbedder(channels[-1], 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [self.base_size_ss] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            grid_pos_emb = pos_embedder(coords)
            self.register_buffer("grid_pos_emb_ss", grid_pos_emb)
            # Token projection layer
            self.token_proj_ss = nn.Linear(channels[-1]*2, channels[-1])

            # Out layers
            self.out_layer = nn.ModuleList(
                [TransformerCrossBlock(
                    channels=channels[-1],
                    ctx_channels=channels[-1]*2,
                    out_channels=channels[-1],
                    num_heads=16,
                    attn_mode="full",
                    qkv_bias=False,
                    x_is_query=False)] + 
                [TransformerBlock(
                    channels=channels[-1],
                    out_channels=channels[-1],
                    num_heads=16,
                    attn_mode="full",
                    qkv_bias=False,
                ) for _ in range(4)] + 
                [FeedForwardNet(
                    channels=channels[-1], 
                    out_channels=latent_channels*2 if not self.latent_denoising else latent_channels)]
            )
        else:
            self.out_layer = nn.Sequential(
                norm_layer(norm_type, channels[-1]),
                nn.SiLU(),
                nn.Conv3d(channels[-1], latent_channels*2 if not self.latent_denoising else latent_channels, 3, padding=1)
            )
        
        if self.encode_global_skl:
            # Initial Tokens and PE
            self.init_tokens = nn.Parameter(torch.zeros(1, global_token_num_skl, channels[-1]))
            pos_embedder = AbsolutePositionEmbedder(channels[-1], 1)
            coords = torch.arange(global_token_num_skl, device=self.device).reshape(-1, 1)
            tokens_pos_emb = pos_embedder(coords)
            self.register_buffer('tokens_pos_emb', tokens_pos_emb)
            # Grids PE
            upsample_factor = 2 ** (len(channels) - 1)
            self.base_size = 64 // upsample_factor
            pos_embedder = AbsolutePositionEmbedder(channels[-1], 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [self.base_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            grid_pos_emb = pos_embedder(coords)
            self.register_buffer("grid_pos_emb", grid_pos_emb)
            # Token projection layer
            self.token_proj = nn.Linear(channels[-1]*2, channels[-1])

            # Out layers
            self.out_layer_skl = nn.ModuleList(
                [TransformerCrossBlock(
                    channels=channels[-1],
                    ctx_channels=channels[-1]*2,
                    out_channels=channels[-1],
                    num_heads=16,
                    attn_mode="full",
                    qkv_bias=False,
                    x_is_query=False)] + 
                [TransformerBlock(
                    channels=channels[-1],
                    out_channels=channels[-1],
                    num_heads=16,
                    attn_mode="full",
                    qkv_bias=False,
                ) for _ in range(4)] + 
                [FeedForwardNet(
                    channels=channels[-1], 
                    out_channels=latent_channels_skl*2 if not self.latent_denoising_skl else latent_channels_skl)]
            )
        else:
            self.out_layer_skl = nn.Sequential(
                norm_layer(norm_type, channels[-1]),
                nn.SiLU(),
                nn.Conv3d(channels[-1], latent_channels_skl*2 if not self.latent_denoising_skl else latent_channels_skl, 3, padding=1)
            )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
        if self.use_pretrain_branch and self.freeze_pretrain_branch:
            # Freeze: self.input_layer, self.blocks, self.middle_block, self.out_layer
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
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.blocks_skl.apply(convert_module_to_f16)
        self.middle_block_skl.apply(convert_module_to_f16)
        if self.encode_global_skl:
            self.token_proj.apply(convert_module_to_f16)
            self.out_layer_skl.apply(convert_module_to_f16)
        if self.encode_global:
            self.token_proj_ss.apply(convert_module_to_f16)
            self.out_layer.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.blocks_skl.apply(convert_module_to_f32)
        self.middle_block_skl.apply(convert_module_to_f32)
        if self.encode_global_skl:
            self.token_proj.apply(convert_module_to_f32)
            self.out_layer_skl.apply(convert_module_to_f32)
        if self.encode_global:
            self.token_proj_ss.apply(convert_module_to_f32)
            self.out_layer.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor, x_skl: torch.Tensor = None, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.type(self.dtype)
        h_skl = self.input_layer_skl(x_skl)
        h_skl = h_skl.type(self.dtype)

        for block, block_skl in zip(self.blocks, self.blocks_skl):
            h_skl = block_skl(h_skl)
            h = block(h)
        h_skl = self.middle_block_skl(h_skl)
        h = self.middle_block(h)

        if self.encode_global:
            B, C, D, H, W = h.shape
            h = h.view(B, C, D*H*W).permute(0, 2, 1)  # B, N, C
            h = torch.cat([h, self.grid_pos_emb_ss[None].expand(B, -1, -1)], dim=-1).type(h.dtype)
            init_tokens = torch.cat([self.init_tokens_ss, self.tokens_pos_emb_ss[None].expand_as(self.init_tokens_ss)], dim=-1).type(h.dtype)
            tokens = self.token_proj_ss(init_tokens.expand(B, -1, -1))
            h = self.out_layer[0](tokens, h)  # B, global_token_num, C
            for layer in self.out_layer[1:]:
                h = layer(h)
            h = h.type(x.dtype)
            if self.latent_denoising:
                if self.normalize_latent:
                    h = nn.functional.normalize(h, dim=-1) * self.normalize_scale
                mean = h
                logvar = torch.zeros_like(h)
            else:
                mean, logvar = h.chunk(2, dim=2)  # B, global_token_num, C
            if sample_posterior and not self.latent_denoising:
                std = torch.exp(0.5 * logvar)
                z = mean + std * torch.randn_like(std)
            else:
                z = mean
        else:
            h = h.type(x.dtype)
            h = self.out_layer(h)
            if self.latent_denoising:
                if self.normalize_latent:
                    h = nn.functional.normalize(h, dim=1) * self.normalize_scale
                mean = h
                logvar = torch.zeros_like(h)
            else:
                mean, logvar = h.chunk(2, dim=1)
            if sample_posterior and not self.latent_denoising:
                std = torch.exp(0.5 * logvar)
                z = mean + std * torch.randn_like(std)
            else:
                z = mean
        
        if self.encode_global_skl:
            B, C, D, H, W = h_skl.shape
            h_skl = h_skl.view(B, C, D*H*W).permute(0, 2, 1)  # B, N, C
            h_skl = torch.cat([h_skl, self.grid_pos_emb[None].expand(B, -1, -1)], dim=-1).type(h_skl.dtype)
            init_tokens = torch.cat([self.init_tokens, self.tokens_pos_emb[None].expand_as(self.init_tokens)], dim=-1).type(h_skl.dtype)
            tokens = self.token_proj(init_tokens.expand(B, -1, -1))
            h_skl = self.out_layer_skl[0](tokens, h_skl)  # B, global_token_num_skl, C
            for layer in self.out_layer_skl[1:]:
                h_skl = layer(h_skl)
            h_skl = h_skl.type(x_skl.dtype)
            if self.latent_denoising_skl:
                if self.normalize_latent_skl:
                    h_skl = nn.functional.normalize(h_skl, dim=-1) * self.normalize_scale
                mean_skl = h_skl
                logvar_skl = torch.zeros_like(h_skl)
            else:
                mean_skl, logvar_skl = h_skl.chunk(2, dim=2)  # B, global_token_num_skl, C
            if sample_posterior and not self.latent_denoising_skl:
                std_skl = torch.exp(0.5 * logvar_skl)
                z_skl = mean_skl + std_skl * torch.randn_like(std_skl)
            else:
                z_skl = mean_skl
        else:
            h_skl = h_skl.type(x_skl.dtype)
            h_skl = self.out_layer_skl(h_skl)
            if self.latent_denoising_skl:
                if self.normalize_latent_skl:
                    h_skl = nn.functional.normalize(h_skl, dim=1) * self.normalize_scale
                mean_skl = h_skl
                logvar_skl = torch.zeros_like(h_skl)
            else:
                mean_skl, logvar_skl = h_skl.chunk(2, dim=1)
            if sample_posterior and not self.latent_denoising_skl:
                std_skl = torch.exp(0.5 * logvar_skl)
                z_skl = mean_skl + std_skl * torch.randn_like(std_skl)
            else:
                z_skl = mean_skl
        
        if self.latent_denoising:
            mean = mean.detach()
        if self.latent_denoising_skl:
            mean_skl = mean_skl.detach()
                
        if return_raw:
            return z, mean, logvar, z_skl, mean_skl, logvar_skl
        return z, z_skl
        

class AniGenSparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).
    
    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """ 
    def __init__(
        self,
        out_channels: int,
        out_channels_skl: int,
        latent_channels: int,
        latent_channels_skl: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        encode_global: bool = False,
        global_token_num: int = 1024,
        encode_global_skl: bool = True,
        global_token_num_skl: int = 1024,
        use_pretrain_branch: bool = True,
        freeze_pretrain_branch: bool = True,
        modules_to_freeze: Optional[List[str]] = ["input_layer", "blocks", "middle_block", "out_layer"],
        normalize_z: bool = False,
        normalize_z_skl: bool = True,
        normalize_scale: float = 1.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_channels_skl = out_channels_skl
        self.latent_channels = latent_channels
        self.latent_channels_skl = latent_channels_skl
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.encode_global = encode_global
        self.global_token_num = global_token_num
        self.encode_global_skl = encode_global_skl
        self.global_token_num_skl = global_token_num_skl
        self.use_pretrain_branch = use_pretrain_branch
        self.freeze_pretrain_branch = freeze_pretrain_branch
        self.normalize_z = normalize_z
        self.normalize_z_skl = normalize_z_skl
        self.normalize_scale = normalize_scale

        if self.encode_global:
            # Initial Grids and PE
            upsample_factor = 2 ** (len(channels) - 1)
            self.base_size_ss = 64 // upsample_factor
            self.init_grids_ss = nn.Parameter(torch.zeros(1, channels[0], self.base_size_ss**3).permute(0, 2, 1).contiguous().clone())  # 1, N, C
            pos_embedder = AbsolutePositionEmbedder(channels[0], 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [self.base_size_ss] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            grid_pos_emb = pos_embedder(coords)
            self.register_buffer("grid_pos_emb_ss", grid_pos_emb)
            # Tokens PE
            pos_embedder = AbsolutePositionEmbedder(channels[0], 1)
            coords = torch.arange(global_token_num, device=self.device).reshape(-1, 1)
            tokens_pos_emb = pos_embedder(coords)
            self.register_buffer('tokens_pos_emb_ss', tokens_pos_emb)
            # Token projection layer
            self.token_proj_ss = nn.Linear(channels[0]*2, channels[0])

            # Input layers
            self.input_layer = nn.ModuleList(
                [TransformerBlock(
                    channels=channels[0] if _ != 0 else latent_channels + channels[0],
                    out_channels=channels[0],
                    num_heads=4 if _ == 0 else 16,
                    attn_mode="full",
                    qkv_bias=False,
                ) for _ in range(4)] +
                [TransformerCrossBlock(
                    channels=channels[0],
                    ctx_channels=channels[0],
                    out_channels=channels[0],
                    num_heads=16,
                    attn_mode="full",
                    qkv_bias=False,
                    x_is_query=False)]
            )
        else:
            self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        if self.encode_global_skl:
            # Initial Grids and PE
            upsample_factor = 2 ** (len(channels) - 1)
            self.base_size = 64 // upsample_factor
            self.init_grids = nn.Parameter(torch.zeros(1, channels[0], self.base_size**3).permute(0, 2, 1).contiguous().clone())  # 1, N, C
            pos_embedder = AbsolutePositionEmbedder(channels[0], 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [self.base_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            grid_pos_emb = pos_embedder(coords)
            self.register_buffer("grid_pos_emb", grid_pos_emb)
            # Tokens PE
            pos_embedder = AbsolutePositionEmbedder(channels[0], 1)
            coords = torch.arange(global_token_num_skl, device=self.device).reshape(-1, 1)
            tokens_pos_emb = pos_embedder(coords)
            self.register_buffer('tokens_pos_emb', tokens_pos_emb)
            # Token projection layer
            self.token_proj = nn.Linear(channels[0]*2, channels[0])

            # Input layers
            self.input_layer_skl = nn.ModuleList(
                [TransformerBlock(
                    channels=channels[0] if _ != 0 else latent_channels_skl + channels[0],
                    out_channels=channels[0],
                    num_heads=4 if _ == 0 else 16,
                    attn_mode="full",
                    qkv_bias=False,
                ) for _ in range(4)] +
                [TransformerCrossBlock(
                    channels=channels[0],
                    ctx_channels=channels[0],
                    out_channels=channels[0],
                    num_heads=16,
                    attn_mode="full",
                    qkv_bias=False,
                    x_is_query=False)]
            )
        else:
            self.input_layer_skl = nn.Conv3d(latent_channels_skl, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])
        self.middle_block_skl = nn.Sequential(*[
            ResBlock3d(channels[0] if _ == 0 else channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        self.blocks_skl = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )
            self.blocks_skl.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks_skl.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], self.out_channels, 3, padding=1)
        )
        self.out_layer_skl = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], self.out_channels_skl, 3, padding=1)
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
        if self.use_pretrain_branch and self.freeze_pretrain_branch:
            # Freeze: self.input_layer, self.blocks, self.middle_block, self.out_layer
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
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.blocks_skl.apply(convert_module_to_f16)
        self.middle_block_skl.apply(convert_module_to_f16)
        if self.encode_global_skl:
            self.token_proj.apply(convert_module_to_f16)
            self.input_layer_skl.apply(convert_module_to_f16)
        if self.encode_global:
            self.token_proj_ss.apply(convert_module_to_f16)
            self.input_layer.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.blocks_skl.apply(convert_module_to_f32)
        self.middle_block_skl.apply(convert_module_to_f32)
        if self.encode_global_skl:
            self.token_proj.apply(convert_module_to_f32)
            self.input_layer_skl.apply(convert_module_to_f32)
        if self.encode_global:
            self.token_proj_ss.apply(convert_module_to_f32)
            self.input_layer.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    
    def forward(self, x: torch.Tensor, x_skl: torch.Tensor) -> torch.Tensor:
        h = F.normalize(x, dim=1) * self.normalize_scale if self.normalize_z else x
        h_skl = F.normalize(x_skl, dim=1) * self.normalize_scale if self.normalize_z_skl else x_skl
        if self.encode_global:
            B, _, _ = h.shape
            h = torch.cat([h, self.tokens_pos_emb_ss[None].expand(B, -1, -1)], dim=-1).type(self.dtype)
            h = h.type(self.dtype)
            for layer in self.input_layer[:-1]:
                h = layer(h)
            init_grids = torch.cat([self.init_grids_ss, self.grid_pos_emb_ss[None].expand_as(self.init_grids_ss)], dim=-1).type(self.dtype)
            grids = self.token_proj_ss(init_grids.expand(B, -1, -1))
            h = self.input_layer[-1](grids, h)  # B, N, C
            h = h.permute(0, 2, 1).view(B, -1, self.base_size, self.base_size, self.base_size)
        else:
            h = self.input_layer(h)
            h = h.type(self.dtype)
        if self.encode_global_skl:
            B, _, _ = h_skl.shape
            h_skl = torch.cat([h_skl, self.tokens_pos_emb[None].expand(B, -1, -1)], dim=-1).type(self.dtype)
            h_skl = h_skl.type(self.dtype)
            for layer in self.input_layer_skl[:-1]:
                h_skl = layer(h_skl)
            init_grids = torch.cat([self.init_grids, self.grid_pos_emb[None].expand_as(self.init_grids)], dim=-1).type(self.dtype)
            grids = self.token_proj(init_grids.expand(B, -1, -1))
            h_skl = self.input_layer_skl[-1](grids, h_skl)  # B, N, C
            h_skl = h_skl.permute(0, 2, 1).view(B, -1, self.base_size, self.base_size, self.base_size)
        else:
            h_skl = self.input_layer_skl(h_skl)
            h_skl = h_skl.type(self.dtype)
        h_skl = self.middle_block_skl(h_skl)
        h = self.middle_block(h)
        for block, block_skl in zip(self.blocks, self.blocks_skl):
            h_skl = block_skl(h_skl)
            h = block(h)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        h_skl = h_skl.type(x.dtype)
        h_skl = self.out_layer_skl(h_skl)
        return h, h_skl
