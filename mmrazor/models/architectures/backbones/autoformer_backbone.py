# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn import build_activation_layer, build_norm_layer
from torch import Tensor

from mmrazor.models.architectures.dynamic_op import (DynamicLinear,
                                                     DynamicMultiheadAttention,
                                                     DynamicPatchEmbed,
                                                     DynamicSequential)
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS


class TransformerEncoderLayer(BaseBackbone):
    """Autoformer block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (List): Ratio of ffn.
        drop_rate (float): The drop path rate of ffn.
            Defaults to 0.
        attn_drop_rate (float): The drop path rate after attention.
            Defaults to 0.
        qkv_bias (bool, optional): Whether keep bias of qkv.
            Defaults to True.
        act_cfg (Dict, optional): The config for acitvation function.
            Defaults to dict(type='GELU').
        norm_cfg (Dict, optional): The config for normalization.
        init_cfg (Dict, optional): The config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: Union[int, BaseMutable],
                 num_heads: Union[int, BaseMutable],
                 mlp_ratio: Union[float, BaseMutable],
                 drop_rate: float,
                 attn_drop_rate: float,
                 qkv_bias: bool = True,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='mmrazor.DynamicLayerNorm'),
                 init_cfg: Dict = None) -> None:
        super().__init__(init_cfg)

        # supernet settings
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = DynamicMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # derived mutable
        self.middle_channels = int(self.embed_dims * self.mlp_ratio)
        self.fc1 = DynamicLinear(embed_dims, self.middle_channels)
        self.fc2 = DynamicLinear(self.middle_channels, embed_dims)
        self.act = build_activation_layer(act_cfg)

    @property
    def norm1(self):
        """The first normalization."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """The second normalization."""
        return getattr(self, self.norm2_name)

    def mutate_encoder_layer(self, mutable_embed_dims: BaseMutable,
                             mutable_num_heads: BaseMutable,
                             mutable_mlp_ratios: BaseMutable):
        """Mutate the mutables of encoder layer."""
        # record the mutables
        self.mutable_embed_dims = mutable_embed_dims
        self.mutable_num_heads = mutable_num_heads
        self.mutable_mlp_ratios = mutable_mlp_ratios
        self.mutable_head_dims = OneShotMutableValue(
            value_list=[64], default_value=64)

        # handle the mutable of the first dynamic LN
        self.norm1.register_mutable_attr('num_features', mutable_embed_dims)

        # handle the mutable in multihead attention
        mutable_q_embed_dims: OneShotMutableValue = mutable_num_heads.derive_expand_mutable(  # noqa: E501
            64)  # noqa: E501
        self.attn.register_mutable_attr('embed_dims', mutable_embed_dims)
        self.attn.register_mutable_attr('num_heads', mutable_num_heads)
        self.attn.register_mutable_attr('q_embed_dims', mutable_q_embed_dims)
        self.attn.rel_pos_embed_k.register_mutable_attr(
            'head_dims', self.mutable_head_dims)
        self.attn.rel_pos_embed_v.register_mutable_attr(
            'head_dims', self.mutable_head_dims)

        # handle the mutable of the second dynamic LN
        self.norm2.register_mutable_attr('num_features', mutable_embed_dims)

        # handle the mutable of FFN
        # mutable channel x mutable value
        self.middle_channels = mutable_embed_dims.derive_expand_mutable(
            mutable_mlp_ratios)
        self.fc1.register_mutable_attr('in_channels', mutable_embed_dims)
        self.fc1.register_mutable_attr('out_channels', self.middle_channels)
        self.fc2.register_mutable_attr('in_channels', self.middle_channels)
        self.fc2.register_mutable_attr('out_channels', mutable_embed_dims)

    def forward(self, x: Tensor) -> Tensor:
        """Forward of Transformer Encode Layer."""
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


@MODELS.register_module()
class AutoformerBackbone(BaseBackbone):
    """Autoformer backbone.

    A PyTorch implementation of Autoformer introduced by:
    `AutoFormer: Searching Transformers for Visual Recognition
    <https://arxiv.org/abs/2107.00651>`_

    Modified from the `official repo
    <https://github.com/microsoft/Cream/blob/main/AutoFormer/>`.

    Args:
        img_size (int, optional): The image size of input.
            Defaults to 224.
        patch_size (int, optional): The patch size of autoformer.
            Defaults to 16.
        in_channels (int, optional): The input channel dimension.
            Defaults to 3.
        qkv_bias (bool, optional): Whether to keep bias of qkv.
            Defaults to True.
        norm_cfg (Dict, optional): The config of normalization.
            Defaults to dict(type='mmrazor.DynamicLayerNorm').
        act_cfg (Dict, optional): The config of activation functions.
            Defaults to dict(type='GELU').
        final_norm (bool, optional): Whether use final normalization.
            Defaults to True.
        drop_out (float, optional): The dropout rate. Defaults to 0..
        init_cfg (Dict, optional): The config for initialization.
            Defaults to None.
    """
    # supernet settings
    arch_settings = {
        'embed_dims': 624,
        'num_heads': 10,
        'mlp_ratios': 4.0,
        'depth': 16,
    }
    # search space
    mutable_settings: Dict[str, List] = {
        'mlp_ratios': [3.0, 3.5, 4.0],  # mutable value
        'num_heads': [8, 9, 10],  # mutable value
        'depth': [14, 15, 16],  # mutable value
        'embed_dims': [528, 576, 624],  # mutable channel
    }

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 qkv_bias: bool = True,
                 norm_cfg: Dict = dict(type='mmrazor.DynamicLayerNorm'),
                 act_cfg: Dict = dict(type='GELU'),
                 final_norm: bool = True,
                 drop_out: float = 0.,
                 init_cfg: Dict = None) -> None:

        super().__init__(init_cfg)

        self.img_size = img_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.in_channels = in_channels
        self.dropout = drop_out
        self.act_cfg = act_cfg

        # supernet settings
        self.embed_dims = int(self.arch_settings['embed_dims'])
        self.num_heads = int(self.arch_settings['num_heads'])
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.depth = self.arch_settings['depth']

        # adapt mutable settings
        self.mlp_ratio_range: List = self.mutable_settings['mlp_ratios']
        self.num_head_range: List = self.mutable_settings['num_heads']
        self.depth_range: List = self.mutable_settings['depth']
        self.embed_dim_range: List = self.mutable_settings['embed_dims']

        # patch embeddings
        self.last_mutable = None
        self.patch_embed = DynamicPatchEmbed(
            img_size=self.img_size,
            in_channels=self.in_channels,
            embed_dims=self.embed_dims)

        # # mutable variables of autoformer
        self.mutable_embed_dims = OneShotMutableChannel(
            num_channels=self.embed_dims,
            candidate_mode='number',
            candidate_choices=self.embed_dim_range)
        self.mutable_depth = OneShotMutableValue(
            value_list=self.depth_range, default_value=self.depth)

        # num of patches
        self.patch_resolution = [
            img_size // patch_size, img_size // patch_size
        ]
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # cls token and pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dims))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # main body
        self.blocks = self.make_layers(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            mlp_ratios=self.mlp_ratios,
            depth=self.depth)

        self.final_norm = final_norm
        if self.final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                      self.embed_dims)
            self.add_module(self.norm1_name, norm1)

        self.mutate()

    @property
    def norm1(self):
        """The first normalization."""
        return getattr(self, self.norm1_name)

    def make_layers(self, embed_dims, num_heads, mlp_ratios, depth):
        """Build multiple TransformerEncoderLayers."""
        layers = []
        for _ in range(depth):
            layer = TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                drop_rate=0.,
                attn_drop_rate=self.dropout,
                act_cfg=self.act_cfg)
            layers.append(layer)
        return DynamicSequential(*layers)

    def mutate(self):
        """Mutate the autoformer."""
        # handle the mutation of depth
        self.blocks.register_mutable_attr('depth', self.mutable_depth)

        # handle the mutation of patch embed
        self.patch_embed.register_mutable_attr(
            'embed_dims', self.mutable_embed_dims.derive_same_mutable())
        self.last_mutable = self.mutable_embed_dims

        # handle the dependencies of TransformerEncoderLayers
        for i in range(self.depth):  # max depth here
            layer = self.blocks[i]

            mutable_num_heads = OneShotMutableValue(
                value_list=self.num_head_range, default_value=self.num_heads)
            mutable_mlp_ratios = OneShotMutableValue(
                value_list=self.mlp_ratio_range, default_value=self.mlp_ratios)

            layer.mutate_encoder_layer(
                mutable_embed_dims=self.last_mutable.derive_same_mutable(),
                mutable_num_heads=mutable_num_heads,
                mutable_mlp_ratios=mutable_mlp_ratios)

        # handle the mutable of final norm
        if self.final_norm:
            self.norm1.register_mutable_attr(
                'num_features', self.last_mutable.derive_same_mutable())

    def forward(self, x: Tensor):
        """Forward of Autoformer."""
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.mutable_embed_dims is not None:
            embed_dims = self.mutable_embed_dims.current_choice
        else:
            embed_dims = self.embed_dims

        # cls token
        cls_tokens = self.cls_token[..., :embed_dims].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # pos embed
        x = x + self.pos_embed[..., :embed_dims]

        x = F.dropout(x, p=self.dropout)

        # dynamic depth
        x = self.blocks(x)

        if self.final_norm:
            x = self.norm1(x)
        return (torch.mean(x[:, 1:], dim=1), )
