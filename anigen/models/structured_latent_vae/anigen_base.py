from typing import *
import torch
import torch.nn as nn
from ...modules import sparse as sp
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules.sparse.transformer import SparseTransformerMultiContextCrossBlock, SparseTransformerBlock
from ...modules.transformer import AbsolutePositionEmbedder, TransformerCrossBlock


class FreqPositionalEmbedder(nn.Module):
    def __init__(self, in_dim, include_input=True, max_freq_log2=8, num_freqs=8, log_sampling=True, periodic_fns=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = None
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns if periodic_fns is not None else [
            torch.sin, torch.cos
        ]
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.in_dim
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

def block_attn_config(self, attn_mode_attr='attn_mode'):
    """
    Return the attention configuration of the model.
    """
    attn_mode = getattr(self, attn_mode_attr)
    for i in range(self.num_blocks):
        if attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif attn_mode == "full":
            yield "full", None, None, None, None
        elif attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class AniGenSparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        in_channels_skl: int,
        in_channels_skin: int,
        model_channels: int,
        model_channels_skl: int,
        model_channels_skin: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_heads_skl: int = 8,
        num_heads_skin: int = 8,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        attn_mode_cross: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,

        skin_cross_from_geo: bool = True,
        skl_cross_from_geo: bool = True,
        skin_skl_cross: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_skl = in_channels_skl
        self.in_channels_skin = in_channels_skin
        self.model_channels = model_channels
        self.model_channels_skl = model_channels_skl
        self.model_channels_skin = model_channels_skin
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.attn_mode_cross = attn_mode_cross
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.skin_cross_from_geo = skin_cross_from_geo
        self.skl_cross_from_geo = skl_cross_from_geo
        self.skin_skl_cross = skin_skl_cross

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
            self.pos_embedder_skl = AbsolutePositionEmbedder(model_channels_skl)
            self.pos_embedder_skin = AbsolutePositionEmbedder(model_channels_skin)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.input_layer_skl = sp.SparseLinear(in_channels_skl, model_channels_skl)
        self.input_layer_skin = sp.SparseLinear(in_channels_skin, model_channels_skin)

        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

        ctx_channels = []
        if skin_skl_cross:
            ctx_channels.append(model_channels_skl)
        if skin_cross_from_geo:
            ctx_channels.append(model_channels)
        self.blocks_skin = nn.ModuleList([
            SparseTransformerMultiContextCrossBlock(
                model_channels_skin,
                ctx_channels=ctx_channels,
                num_heads=num_heads_skin,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                attn_mode_cross=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                cross_attn_cache_suffix='_skin',
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self, "attn_mode_cross")
        ])

        ctx_channels = []
        if skin_skl_cross:
            ctx_channels.append(model_channels_skin)
        if skl_cross_from_geo:
            ctx_channels.append(model_channels)
        self.blocks_skl = nn.ModuleList([
            SparseTransformerMultiContextCrossBlock(
                model_channels_skl,
                ctx_channels=ctx_channels,
                num_heads=num_heads_skl,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                attn_mode_cross=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                cross_attn_cache_suffix='_skl',
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self, "attn_mode_cross")
        ])

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
        self.blocks_skin.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        self.blocks_skl.apply(convert_module_to_f32)
        self.blocks_skin.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward_input_layer(self, x: sp.SparseTensor, layer, pos_embedder) -> sp.SparseTensor:
        h = layer(x)
        if self.pe_mode == "ape":
            h = h + pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)
        return h

    def forward(self, x: sp.SparseTensor, x_skl: sp.SparseTensor, x_skin: sp.SparseTensor) -> sp.SparseTensor:
        h = self.forward_input_layer(x, self.input_layer, self.pos_embedder)
        h_skl = self.forward_input_layer(x_skl, self.input_layer_skl, self.pos_embedder_skl)
        h_skin = self.forward_input_layer(x_skin, self.input_layer_skin, self.pos_embedder_skin)

        for block, block_skl, block_skin in zip(self.blocks, self.blocks_skl, self.blocks_skin):
            f, f_skl, f_skin = h, h_skl, h_skin
            h = block(f)
            skl_contexts, skin_contexts = [], []
            if self.skin_skl_cross:
                skl_contexts.append(f_skin)
                skin_contexts.append(f_skl)
            if self.skl_cross_from_geo:
                skl_contexts.append(f)
            if self.skin_cross_from_geo:
                skin_contexts.append(f)
            h_skl = block_skl(f_skl, skl_contexts)
            h_skin = block_skin(f_skin, skin_contexts)
        return h, h_skl, h_skin
