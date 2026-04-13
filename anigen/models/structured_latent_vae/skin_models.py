import torch
import torch.nn as nn
from typing import *
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from ...modules.transformer import TransformerBlock, FeedForwardNet
from .anigen_base import FreqPositionalEmbedder, TransformerCrossBlock
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


class Embedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=None, depth: int = 4, mlp_ratio: float = 4.0, jp_embed_attn: bool = True):
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.jp_embed_attn = jp_embed_attn
        self.in_layer = FeedForwardNet(channels=in_dim, out_channels=hidden_dim, mlp_ratio=mlp_ratio)
        if self.jp_embed_attn:
            self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads=8, attn_mode='full') for _ in range(depth)])
            for block in self.blocks:
                block.to(torch.float16)
        else:
            self.blocks = nn.ModuleList([FeedForwardNet(channels=hidden_dim, out_channels=hidden_dim, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.out_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        h = x
        for block in self.blocks:
            h = block(h[None].type(torch.float16))[0] if self.jp_embed_attn else block(h) + x
        h = self.out_layer(h.type(x.dtype))
        return h


class SkinEncoder(nn.Module):
    def __init__(self, skin_feat_channels: int = 8, skl_pos_embed_freq: int = 10, jp_embedder_config: Optional[Dict[str, Any]] = {}, jp_embed_dim: int = 128, relative_pe=True, vert_feat_is_linear=True, normalize_feat=True, **kwargs):
        super().__init__()
        self.skin_feat_channels = skin_feat_channels
        self.skl_pos_embed_freq = skl_pos_embed_freq
        self.jp_embedder_config = jp_embedder_config

        self.pos_embedder_fourier = FreqPositionalEmbedder(in_dim=3, max_freq_log2=self.skl_pos_embed_freq, num_freqs=self.skl_pos_embed_freq, include_input=True)
        self.pos_embedder_linear = nn.Linear(self.pos_embedder_fourier.out_dim, jp_embed_dim)
        self.root_embedding = nn.Parameter(torch.zeros(1, jp_embed_dim))
        self.joint_embedder = Embedder(in_dim=2 * jp_embed_dim, out_dim=jp_embed_dim, **self.jp_embedder_config)
        self.out_layer_vert = FeedForwardNet(channels=jp_embed_dim, out_channels=skin_feat_channels)
        self.out_layer_joint = FeedForwardNet(channels=jp_embed_dim, out_channels=skin_feat_channels)
        self.relative_pe = relative_pe
        self.vert_feat_is_linear = vert_feat_is_linear
        self.normalize_feat = normalize_feat

    def forward(self, joints_list: List[torch.Tensor], parents_list: List[torch.Tensor], skin_list: List[torch.Tensor]=None):
        vert_skin_embeds = [] if skin_list is not None else None
        joint_skin_embeds = []
        for i in range(len(joints_list)):
            parent_idx = parents_list[i].clone()
            joints = joints_list[i]
            if self.relative_pe:
                joints = joints - torch.cat([joints, joints[:1]])[parent_idx]
            joints_pos_embed = self.pos_embedder_linear(self.pos_embedder_fourier(joints))
            joints_pos_embed = torch.cat([joints_pos_embed, self.root_embedding], dim=0)
            parents_pos_embed = joints_pos_embed[parent_idx]
            jp_pos_embed = torch.cat([joints_pos_embed[:-1], parents_pos_embed], dim=-1)
            joints_embed = self.joint_embedder(jp_pos_embed)
            if self.normalize_feat:
                joints_embed = torch.nn.functional.normalize(joints_embed, dim=-1)
            if skin_list is not None:
                vert_skin = skin_list[i]
                if self.vert_feat_is_linear:
                    joints_embed_for_vert = self.out_layer_vert(joints_embed)
                    vert_skin_embed = vert_skin @ joints_embed_for_vert
                else:
                    vert_skin_embed = vert_skin @ joints_embed
                    vert_skin_embed = self.out_layer_vert(vert_skin_embed)
                vert_skin_embeds.append(vert_skin_embed)
            joints_embed = self.out_layer_joint(joints_embed)
            joint_skin_embeds.append(joints_embed)
        return joint_skin_embeds, vert_skin_embeds


def clamp_with_grad(x, min, max):
    return x + (x.clamp(min, max) - x).detach()


class TreeTransformerSkinDecoder(nn.Module):
    # The principles of the tree transformer skinning model:
    # 1. joint features are related to the tree structure, since the decoding process is skeleton-agnostic.
    # 2. decode the skinning weights directly, hoping the transformer can handle the skinning assignment. 
    # It's a pure learning-based method.
    def __init__(self, 
                 skin_feat_channels: int, 
                 model_channels: int=512, 
                 num_heads=4, 
                 num_blocks=4,
                 vert_cross_blocks_num: int = 1,
                 use_fp16: bool = False):
        super().__init__()
        self.skin_feat_channels = skin_feat_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.root_features = nn.Parameter(torch.zeros([1, skin_feat_channels]), requires_grad=True)
        self.input_layer_vertex = nn.Linear(skin_feat_channels, model_channels)
        self.input_layer_skin   = nn.Linear(skin_feat_channels*2, model_channels)
        assert vert_cross_blocks_num <= num_blocks, f"vert_cross_blocks_num should be less than or equal to num_blocks, got {vert_cross_blocks_num} and {num_blocks}."
        self.vert_cross_blocks_num = vert_cross_blocks_num
        self.blocks_vertex = nn.ModuleList([
            TransformerCrossBlock(
                channels=model_channels,
                ctx_channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=4.0,
                attn_mode="full",
                no_self=True)
            for _ in range(self.vert_cross_blocks_num)
        ] + [
            FeedForwardNet(
                channels=model_channels,
                mlp_ratio=4.0,
                out_channels=model_channels,
            )
            for _ in range(num_blocks - self.vert_cross_blocks_num)
        ])
        self.blocks_skin = nn.ModuleList([
            TransformerBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=4.0,
                attn_mode="full")
            for _ in range(num_blocks)
        ])
        self.out_layer_vertex = nn.Sequential(
            nn.Linear(model_channels, model_channels*4),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_channels*4, model_channels+1),
        )
        self.out_layer_skin = nn.Sequential(
            nn.Linear(model_channels, model_channels*4),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_channels*4, model_channels),
        )
        self.temp_activation = nn.ELU(alpha=1.0)
        self.dtype = torch.float16 if use_fp16 else torch.float32

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
        self.blocks_vertex.apply(convert_module_to_f16)
        self.blocks_skin.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks_vertex.apply(convert_module_to_f32)
        self.blocks_skin.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, vertex_features, joint_features, parents) -> torch.Tensor:
        j_num = joint_features.shape[1]
        h_v = vertex_features
        h_v =  self.input_layer_vertex(h_v)
        h_j = joint_features
        h_j = torch.cat([h_j, self.root_features[None]], dim=1)
        parents = torch.where(parents < 0, torch.ones_like(parents)*j_num, parents)
        h_j = torch.cat([h_j[:, :-1], h_j[:, parents[0]]], dim=-1)
        h_j = self.input_layer_skin(h_j)
        h_v = h_v.type(self.dtype)
        h_j = h_j.type(self.dtype)
        blocks_num = len(self.blocks_vertex)
        for idx, block_v, block_j in zip(range(blocks_num), self.blocks_vertex, self.blocks_skin):
            f_v, f_j = h_v, h_j
            h_v = block_v(f_v, f_j) if idx < self.vert_cross_blocks_num else block_v(f_v)
            h_j = block_j(f_j)
        h_v = h_v.type(vertex_features.dtype)
        h_j = h_j.type(joint_features.dtype)
        h_v = self.out_layer_vertex(h_v)
        h_j = self.out_layer_skin(h_j)
        h_v, inv_temp = h_v[..., :-1], h_v[..., -1].unsqueeze(-1)
        inv_temp = self.temp_activation(inv_temp) + self.temp_activation.alpha + 1.0
        skin_weights = torch.einsum("nac,nbc->nab", h_v, h_j)
        skin_weights = torch.softmax(skin_weights * inv_temp, dim=-1)
        return skin_weights


SKIN_MODEL_DICT = {'tree': TreeTransformerSkinDecoder}


class SkinAutoEncoder(nn.Module):
    def __init__(self, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any], use_fp16: bool = False):
        super().__init__()
        self.skin_encoder = SkinEncoder(**encoder_config)
        decoder_config['use_fp16'] = use_fp16
        self.skin_decoder = SKIN_MODEL_DICT[decoder_config.pop('model_type')](**decoder_config)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()
    
    def convert_to_fp16(self) -> None:
        self.skin_decoder.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        self.skin_decoder.convert_to_fp32()
    
    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def encode(self, joints_list: List[torch.Tensor], parents_list: List[torch.Tensor], skin_list: List[torch.Tensor]):
        joint_skin_embeds, vert_skin_embeds = self.skin_encoder(joints_list, parents_list, skin_list)
        return joint_skin_embeds, vert_skin_embeds

    def decode(self, vertex_features, joint_features, parents) -> torch.Tensor:
        skin_weights = self.skin_decoder(vertex_features, joint_features, parents)
        return skin_weights
    
    def forward(self, joints_list: List[torch.Tensor], parents_list: List[torch.Tensor], skin_list: List[torch.Tensor]):
        joint_skin_embeds, vert_skin_embeds = self.skin_encoder(joints_list, parents_list, skin_list)
        skin_pred_list = []
        for i in range(len(joints_list)):
            skin_pred = self.skin_decoder(vert_skin_embeds[i][None], joint_skin_embeds[i][None], parents_list[i][None])
            skin_pred_list.append(skin_pred[0])
        return skin_pred_list, joint_skin_embeds, vert_skin_embeds


class AniGenElasticSLatEncoderGamma(SparseTransformerElasticMixin, SkinAutoEncoder):
    """
    SLat VAE encoder with elastic memory management.
    Used for training with low VRAM.
    """
