# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, no_type_check

import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_op.bricks import DynamicSequential
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.ops import MBBlock
from mmrazor.registry import MODELS


def _range_to_list(range_: List[int]) -> List[int]:
    assert len(range_) == 3

    start, end, step = range_
    return list(range(start, end + 1, step))


def _mutate_conv_module(
        conv_module: ConvModule,
        mutable_in_channels: Optional[BaseMutable] = None,
        mutable_out_channels: Optional[BaseMutable] = None,
        mutable_kernel_size: Optional[Tuple[BaseMutable,
                                            Sequence[int]]] = None):
    if mutable_in_channels is not None:
        conv_module.conv.mutate_in_channels(mutable_in_channels)
    if mutable_out_channels is not None:
        conv_module.conv.mutate_out_channels(mutable_out_channels)
        if conv_module.with_norm:
            conv_module.bn.mutate_num_features(mutable_out_channels)

    if mutable_kernel_size is not None:
        mutable_kernel_size, kernel_size_list = mutable_kernel_size
        if mutable_kernel_size is not None:
            conv_module.conv.mutate_kernel_size(
                mutable_kernel_size.derive_same_mutable(), kernel_size_list)


@MODELS.register_module()
class BigNASMobileNet(BaseBackbone):
    # Parameters to build layers. 5 parameters are needed to construct a
    # layer, from left to right:
    # expand_ratio, channels, num_blocks, kernel_size, stride
    arch_settings = [
        [1, 24, 2, 3, 1],
        [6, 32, 3, 3, 2],
        [6, 48, 3, 5, 2],
        [6, 88, 4, 5, 2],
        [6, 128, 6, 5, 1],
        [6, 216, 6, 5, 2],
        [6, 352, 2, 5, 1],
    ]

    # [min_channels, max_channels, step]
    # [min_num_blocks, max_num_blocks, step]
    # [min_kernel_size, max_kernel_size, step]
    mutable_settings = [[[32, 40, 8], None, None],
                        [[16, 24, 8], [1, 2, 1], None],
                        [[24, 32, 8], [2, 3, 1], None],
                        [[40, 48, 8], [2, 3, 1], [3, 5, 2]],
                        [[80, 88, 8], [2, 4, 1], [3, 5, 2]],
                        [[112, 128, 8], [2, 6, 1], [3, 5, 2]],
                        [[192, 216, 8], [2, 6, 1], [3, 5, 2]],
                        [[320, 352, 8], [1, 2, 1], [3, 5, 2]],
                        [[1280, 1408, 8], None, None]]

    def __init__(self,
                 first_out_channels=40,
                 last_out_channels=1408,
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

        first_out_channels = make_divisible(first_out_channels * widen_factor,
                                            8)
        self.in_channels = first_out_channels

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            print(f'stage {i + 1}: {layer_cfg}')
            expand_ratio, channels, num_blocks, \
                kernel_size, stride = layer_cfg
            out_channels = make_divisible(channels * widen_factor, 8)

            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(last_out_channels * widen_factor)
        else:
            self.out_channel = last_out_channels
        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

        self.all_layers = ['conv1'] + self.layers
        self.mutate()

    def make_layer(self, out_channels, num_blocks, kernel_size, stride,
                   expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
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
            layers.append(mb_layer)
            self.in_channels = out_channels

        return DynamicSequential(*layers)

    # FIXME
    # fix type lint
    @no_type_check
    def mutate(self) -> None:
        source_mutables = nn.ModuleDict()

        last_mutable = None
        for layer_idx, mutable_cfg in enumerate(self.mutable_settings):
            layer = getattr(self, self.all_layers[layer_idx])
            mutable_list = nn.ModuleList()

            channels_range: Optional[List] = mutable_cfg[0]
            num_blocks_range: Optional[List] = mutable_cfg[1]
            kernel_size_range: Optional[List] = mutable_cfg[2]
            channels_list: Optional[List] = None
            num_blocks_list: Optional[List[int]] = None
            kernel_size_list: Optional[List[int]] = None
            mutable_channel: Optional[BaseMutable] = None
            mutable_depth: Optional[BaseMutable] = None
            mutable_kernel_size: Optional[BaseMutable] = None
            if channels_range is not None:
                channels_list = _range_to_list(channels_range)
                mutable_channel = OneShotMutableChannel(
                    num_channels=max(channels_list),
                    candidate_mode='number',
                    candidate_choices=channels_list)
                mutable_list.append(mutable_channel)
            if num_blocks_range is not None:
                num_blocks_list = _range_to_list(num_blocks_range)
                mutable_depth = OneShotMutableValue(
                    value_list=num_blocks_list,
                    default_value=max(num_blocks_list))
                mutable_list.append(mutable_depth)
            if kernel_size_range is not None:
                kernel_size_list = _range_to_list(kernel_size_range)
                mutable_kernel_size = OneShotMutableValue(
                    value_list=kernel_size_list,
                    default_value=max(kernel_size_list))
                mutable_list.append(mutable_kernel_size)
            source_mutables[self.all_layers[layer_idx]] = mutable_list

            if layer_idx == 0 or layer_idx == len(self.all_layers) - 1:
                if last_mutable is None:
                    derive_in_channels = None
                else:
                    last_mutable = last_mutable * 1
                _mutate_conv_module(
                    layer,
                    mutable_in_channels=derive_in_channels,
                    mutable_out_channels=mutable_channel * 1)
                last_mutable = mutable_channel
            else:
                if mutable_depth is not None:
                    layer.mutate_depth(mutable_depth, num_blocks_list)
                for mb_layer in layer:
                    # HACK
                    # try modify __iter__ ?
                    if isinstance(mb_layer, layer.forward_ignored_module):
                        continue
                    expand_ratio = mb_layer.expand_ratio
                    if mb_layer.with_expand_conv:
                        _mutate_conv_module(
                            mb_layer.expand_conv,
                            mutable_in_channels=last_mutable * 1,
                            mutable_out_channels=last_mutable * expand_ratio)
                    _mutate_conv_module(
                        mb_layer.depthwise_conv,
                        mutable_in_channels=last_mutable * expand_ratio,
                        mutable_out_channels=last_mutable * expand_ratio,
                        mutable_kernel_size=(mutable_kernel_size,
                                             kernel_size_list))
                    if mb_layer.with_res_shortcut:
                        mutable_out_channels = last_mutable * 1
                    else:
                        mutable_out_channels = mutable_channel * 1
                    _mutate_conv_module(
                        mb_layer.linear_conv,
                        mutable_in_channels=last_mutable * expand_ratio,
                        mutable_out_channels=mutable_out_channels)
                    if not mb_layer.with_res_shortcut:
                        last_mutable = mutable_channel

        self.last_mutable = last_mutable
        self.source_mutables = source_mutables

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
