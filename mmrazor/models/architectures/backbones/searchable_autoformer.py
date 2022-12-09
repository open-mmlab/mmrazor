# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicLinear, DynamicMultiheadAttention, DynamicPatchEmbed,
    DynamicSequential)
from mmrazor.models.mutables import (BaseMutable, BaseMutableChannel,
                                     MutableChannelContainer,
                                     OneShotMutableChannel,
                                     OneShotMutableValue)
from mmrazor.models.mutables.mutable_channel import OneShotMutableChannelUnit
from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')


class TransformerEncoderLayer(BaseBackbone):
    """Autoformer block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (List): Ratio of ffn.
        attn_drop_rate (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop_rate (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        out_drop_rate (dict): Dropout rate of the dropout layer before adding
            the shortcut. Defaults to 0.
        qkv_bias (bool, optional): Whether to keep bias of qkv.
            Defaults to True.
        act_cfg (Dict, optional): The config for acitvation function.
            Defaults to dict(type='GELU').
        norm_cfg (Dict, optional): The config for normalization.
            Defaults to dict(type='mmrazor.DynamicLayerNorm').
        init_cfg (Dict, optional): The config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: float,
                 proj_drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 out_drop_rate: float = 0.,
                 qkv_bias: bool = True,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='mmrazor.DynamicLayerNorm'),
                 init_cfg: Dict = None) -> None:
        super().__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = DynamicMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            out_drop_rate=out_drop_rate,
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        middle_channels = int(embed_dims * mlp_ratio)
        self.fc1 = DynamicLinear(embed_dims, middle_channels)
        self.fc2 = DynamicLinear(middle_channels, embed_dims)
        self.act = build_activation_layer(act_cfg)

    @property
    def norm1(self):
        """The first normalization."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """The second normalization."""
        return getattr(self, self.norm2_name)

    def register_mutables(self, mutable_num_heads: BaseMutable,
                          mutable_mlp_ratios: BaseMutable,
                          mutable_q_embed_dims: BaseMutable,
                          mutable_head_dims: BaseMutable,
                          mutable_embed_dims: BaseMutable):
        """Mutate the mutables of encoder layer."""
        # record the mutables
        self.mutable_num_heads = mutable_num_heads
        self.mutable_mlp_ratios = mutable_mlp_ratios
        self.mutable_q_embed_dims = mutable_q_embed_dims
        self.mutable_embed_dims = mutable_embed_dims
        self.mutable_head_dims = mutable_head_dims
        # handle the mutable of FFN
        self.middle_channels = mutable_mlp_ratios * mutable_embed_dims

        self.attn.register_mutable_attr('num_heads', mutable_num_heads)

        # handle the mutable of the first dynamic LN
        MutableChannelContainer.register_mutable_channel_to_module(
            self.norm1, self.mutable_embed_dims, True)
        # handle the mutable of the second dynamic LN
        MutableChannelContainer.register_mutable_channel_to_module(
            self.norm2, self.mutable_embed_dims, True)

        # handle the mutable of attn
        MutableChannelContainer.register_mutable_channel_to_module(
            self.attn, self.mutable_embed_dims, False)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.attn,
            self.mutable_q_embed_dims,
            True,
            end=self.mutable_q_embed_dims.current_choice)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.attn.rel_pos_embed_k, self.mutable_head_dims, False)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.attn.rel_pos_embed_v, self.mutable_head_dims, False)

        # handle the mutable of fc
        MutableChannelContainer.register_mutable_channel_to_module(
            self.fc1, mutable_embed_dims, False)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.fc1,
            self.middle_channels,
            True,
            start=0,
            end=self.middle_channels.current_choice)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.fc2,
            self.middle_channels,
            False,
            start=0,
            end=self.middle_channels.current_choice)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.fc2, mutable_embed_dims, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        arch_setting (Dict[str, List]): Architecture settings.
        img_size (int, optional): The image size of input.
            Defaults to 224.
        patch_size (int, optional): The patch size of autoformer.
            Defaults to 16.
        in_channels (int, optional): The input channel dimension.
            Defaults to 3.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool, optional): Whether to keep bias of qkv.
            Defaults to True.
        norm_cfg (Dict, optional): The config of normalization.
            Defaults to dict(type='mmrazor.DynamicLayerNorm').
        act_cfg (Dict, optional): The config of activation functions.
            Defaults to dict(type='GELU').
        use_final_norm (bool, optional): Whether use final normalization.
            Defaults to True.
        init_cfg (Dict, optional): The config for initialization.
            Defaults to None.

    Excamples:
        >>> arch_setting = dict(
        ...     mlp_ratios=[3.0, 3.5, 4.0],
        ...     num_heads=[8, 9, 10],
        ...     depth=[14, 15, 16],
        ...     embed_dims=[528, 576, 624]
        ... )
        >>> model = AutoformerBackbone(arch_setting=arch_setting)
    """

    def __init__(self,
                 arch_setting: Dict[str, List],
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 norm_cfg: Dict = dict(type='mmrazor.DynamicLayerNorm'),
                 act_cfg: Dict = dict(type='GELU'),
                 use_final_norm: bool = True,
                 init_cfg: Dict = None) -> None:

        super().__init__(init_cfg)

        self.arch_setting = arch_setting
        self.img_size = img_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.in_channels = in_channels
        self.drop_rate = drop_rate
        self.use_final_norm = use_final_norm
        self.act_cfg = act_cfg

        # adapt mutable settings
        self.mlp_ratio_range: List = self.arch_setting['mlp_ratios']
        self.num_head_range: List = self.arch_setting['num_heads']
        self.depth_range: List = self.arch_setting['depth']
        self.embed_dim_range: List = self.arch_setting['embed_dims']

        # mutable variables of autoformer
        self.mutable_depth = OneShotMutableValue(
            value_list=self.depth_range, default_value=self.depth_range[-1])

        self.mutable_embed_dims = OneShotMutableChannel(
            num_channels=self.embed_dim_range[-1],
            candidate_choices=self.embed_dim_range)

        # handle the mutable in multihead attention
        self.base_embed_dims = OneShotMutableChannel(
            num_channels=64, candidate_choices=[64])

        self.mutable_num_heads = [
            OneShotMutableValue(
                value_list=self.num_head_range,
                default_value=self.num_head_range[-1])
            for _ in range(self.depth_range[-1])
        ]
        self.mutable_mlp_ratios = [
            OneShotMutableValue(
                value_list=self.mlp_ratio_range,
                default_value=self.mlp_ratio_range[-1])
            for _ in range(self.depth_range[-1])
        ]

        self.mutable_q_embed_dims = [
            i * self.base_embed_dims for i in self.mutable_num_heads
        ]

        # patch embeddings
        self.patch_embed = DynamicPatchEmbed(
            img_size=self.img_size,
            in_channels=self.in_channels,
            embed_dims=self.mutable_embed_dims.num_channels)

        # num of patches
        self.patch_resolution = [
            img_size // patch_size, img_size // patch_size
        ]
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # cls token and pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1,
                        self.mutable_embed_dims.num_channels))

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.mutable_embed_dims.num_channels))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        self.dpr = np.linspace(0, drop_path_rate,
                               self.mutable_depth.max_choice)

        # main body
        self.blocks = self._make_layer(
            embed_dims=self.mutable_embed_dims.num_channels,
            depth=self.mutable_depth.max_choice)

        # final norm
        if self.use_final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.mutable_embed_dims.num_channels)
            self.add_module(self.norm1_name, norm1)

        self.last_mutable = self.mutable_embed_dims

        self.register_mutables()

    @property
    def norm1(self):
        """The first normalization."""
        return getattr(self, self.norm1_name)

    def _make_layer(self, embed_dims, depth):
        """Build multiple TransformerEncoderLayers."""
        layers = []
        for i in range(depth):
            layer = TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=self.mutable_num_heads[i].max_choice,
                mlp_ratio=self.mutable_mlp_ratios[i].max_choice,
                proj_drop_rate=self.drop_rate,
                out_drop_rate=self.dpr[i],
                qkv_bias=self.qkv_bias,
                act_cfg=self.act_cfg)
            layers.append(layer)
        return DynamicSequential(*layers)

    def register_mutables(self):
        """Mutate the autoformer."""
        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        # handle the mutation of depth
        self.blocks.register_mutable_attr('depth', self.mutable_depth)

        # handle the mutation of patch embed
        MutableChannelContainer.register_mutable_channel_to_module(
            self.patch_embed, self.mutable_embed_dims, True)

        # handle the dependencies of TransformerEncoderLayers
        for i in range(self.mutable_depth.max_choice):  # max depth here
            layer = self.blocks[i]
            layer.register_mutables(
                mutable_num_heads=self.mutable_num_heads[i],
                mutable_mlp_ratios=self.mutable_mlp_ratios[i],
                mutable_q_embed_dims=self.mutable_q_embed_dims[i],
                mutable_head_dims=self.base_embed_dims,
                mutable_embed_dims=self.last_mutable)

        # handle the mutable of final norm
        if self.use_final_norm:
            MutableChannelContainer.register_mutable_channel_to_module(
                self.norm1, self.last_mutable, True)

    def forward(self, x: torch.Tensor):
        """Forward of Autoformer."""
        B = x.shape[0]
        x = self.patch_embed(x)

        embed_dims = int(self.mutable_embed_dims.current_choice) if isinstance(
            self.mutable_embed_dims,
            BaseMutableChannel) else self.embed_dim_range[-1]

        # cls token
        cls_tokens = self.cls_token[..., :embed_dims].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # pos embed
        x = x + self.pos_embed[..., :embed_dims]
        x = self.drop_after_pos(x)

        # dynamic depth
        x = self.blocks(x)

        if self.use_final_norm:
            x = self.norm1(x)

        return (torch.mean(x[:, 1:], dim=1), )
