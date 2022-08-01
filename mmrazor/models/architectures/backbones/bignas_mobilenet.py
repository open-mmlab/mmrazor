# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import make_divisible
from mmcls.registry import MODELS
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_op import DynamicSequential
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.ops import MBBlock


def _range_to_list(range_: List[int]) -> List[int]:
    assert len(range_) == 3

    start, end, step = range_
    return list(range(start, end + 1, step))


def _mutate_conv_module(
    conv_module: ConvModule,
    in_channels_derive_cfg: Optional[Tuple[BaseMutable, int]] = None,
    out_channels_derive_cfg: Optional[Tuple[BaseMutable, int]] = None,
    kernel_size_derive_cfg: Optional[Tuple[BaseMutable,
                                           Sequence[int]]] = None):
    if in_channels_derive_cfg is not None:
        mutable_in_channels, expand_ratio = in_channels_derive_cfg
        conv_module.conv.mutate_in_channels(
            mutable_in_channels.derive_expand_mutable(expand_ratio))
    if out_channels_derive_cfg is not None:
        mutable_out_channels, expand_ratio = out_channels_derive_cfg
        conv_module.conv.mutate_out_channels(
            mutable_out_channels.derive_expand_mutable(expand_ratio))
        if conv_module.with_norm:
            conv_module.bn.mutate_num_features(
                mutable_out_channels.derive_expand_mutable(expand_ratio))

    if kernel_size_derive_cfg is not None:
        mutable_kernel_size, kernel_size_list = kernel_size_derive_cfg
        conv_module.conv.mutate_kernel_size(
            mutable_kernel_size.derive_same_mutable(), kernel_size_list)


@MODELS.register_module()
class BigNASMobileNet(BaseBackbone):
    # Parameters to build layers. 5 parameters are needed to construct a
    # layer, from left to right:
    # expand_ratio,
    # [min_channel, max_channel, step]
    # [min_num_blocks, max_num_blocks, step]
    # [min_kernel_size, max_kernel_size, step]
    # stride
    arch_settings = [
        [1, [16, 24, 8], [1, 2, 1], [3, 3, 2], 1],
        [6, [24, 32, 8], [2, 3, 1], [3, 3, 2], 2],
        [6, [40, 48, 8], [2, 3, 1], [3, 5, 2], 2],
        [6, [80, 88, 8], [2, 4, 1], [3, 5, 2], 2],
        [6, [112, 128, 8], [2, 6, 1], [3, 5, 2], 1],
        [6, [192, 216, 8], [2, 6, 1], [3, 5, 2], 2],
        [6, [320, 352, 8], [1, 2, 1], [3, 5, 2], 1],
    ]

    def __init__(self,
                 first_out_channels_range=[32, 40, 8],
                 last_out_channels_range=[1280, 1408, 8],
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=dict(type='CenterCropDynamicConv2d'),
                 norm_cfg=dict(type='DynamicBatchNorm2d'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super().__init__(init_cfg)
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.source_mutables = nn.ModuleDict()

        first_out_channels = [
            make_divisible(x * widen_factor, 8)
            for x in _range_to_list(first_out_channels_range)
        ]
        self.in_channels = max(first_out_channels)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        mutable_out_channels = OneShotMutableChannel(
            num_channels=self.in_channels,
            candidate_choices=first_out_channels,
            candidate_mode='number')
        self.source_mutables['conv1'] = mutable_out_channels
        _mutate_conv_module(
            self.conv1, out_channels_derive_cfg=(mutable_out_channels, 1))
        self.last_mutable = mutable_out_channels

        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            print(f'stage {i + 1}: {layer_cfg}')
            expand_ratio, channel_range, num_blocks_range, \
                kernel_size_range, stride = layer_cfg
            out_channels_list = [
                make_divisible(x * widen_factor, 8)
                for x in _range_to_list(channel_range)
            ]
            num_blocks_list = _range_to_list(num_blocks_range)
            kernel_size_list = _range_to_list(kernel_size_range)

            inverted_res_layer = self.make_layer(
                out_channels_list=out_channels_list,
                num_blocks_list=num_blocks_list,
                kernel_size_list=kernel_size_list,
                stride=stride,
                expand_ratio=expand_ratio,
                stage_idx=i + 1)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        last_out_channels = [
            make_divisible(x * widen_factor, 8)
            for x in _range_to_list(last_out_channels_range)
        ]
        out_channels = max(last_out_channels)

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

        mutable_out_channels = OneShotMutableChannel(
            num_channels=out_channels,
            candidate_choices=last_out_channels,
            candidate_mode='number')
        _mutate_conv_module(
            self.conv2,
            in_channels_derive_cfg=(self.last_mutable, 1),
            out_channels_derive_cfg=(mutable_out_channels, 1))
        self.source_mutables['conv2'] = mutable_out_channels
        self.last_mutable = mutable_out_channels

    def make_layer(self, out_channels_list, num_blocks_list, kernel_size_list,
                   stride, expand_ratio, stage_idx):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        mutable_list = nn.ModuleList()
        mutable_kernel_size = OneShotMutableValue(
            value_list=kernel_size_list, default_value=max(kernel_size_list))
        mutable_list.append(mutable_kernel_size)
        last_mutable = self.last_mutable

        out_channels = max(out_channels_list)
        kernel_size = max(kernel_size_list)
        layers = []
        for i in range(max(num_blocks_list)):
            if i >= 1:
                stride = 1
            mb_layer = MBBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp)
            if mb_layer.with_expand_conv:
                _mutate_conv_module(
                    mb_layer.expand_conv,
                    in_channels_derive_cfg=(last_mutable, 1),
                    out_channels_derive_cfg=(last_mutable, expand_ratio))

            _mutate_conv_module(
                mb_layer.depthwise_conv,
                in_channels_derive_cfg=(last_mutable, expand_ratio),
                out_channels_derive_cfg=(last_mutable, expand_ratio),
                kernel_size_derive_cfg=(mutable_kernel_size, kernel_size_list))
            if mb_layer.with_res_shortcut:
                out_channels_derive_cfg = (last_mutable, 1)
            else:
                mutable_out_channels = OneShotMutableChannel(
                    num_channels=out_channels,
                    candidate_choices=out_channels_list,
                    candidate_mode='number')
                out_channels_derive_cfg = (mutable_out_channels, 1)
                mutable_list.append(mutable_out_channels)
            _mutate_conv_module(
                mb_layer.linear_conv,
                in_channels_derive_cfg=(last_mutable, expand_ratio),
                out_channels_derive_cfg=out_channels_derive_cfg)

            layers.append(mb_layer)
            self.in_channels = out_channels

            if not mb_layer.with_res_shortcut:
                last_mutable = mutable_out_channels

        dynamic_seq = DynamicSequential(*layers)
        mutable_depth = OneShotMutableValue(
            value_list=num_blocks_list, default_value=max(num_blocks_list))
        mutable_list.append(mutable_depth)
        dynamic_seq.mutate_depth(mutable_depth, depth_seq=num_blocks_list)

        self.source_mutables[f'stage_{stage_idx}'] = mutable_list
        self.last_mutable = last_mutable

        return dynamic_seq

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
