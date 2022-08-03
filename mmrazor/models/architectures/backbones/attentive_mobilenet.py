# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import make_divisible
from mmcls.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_op import DynamicSequential
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.ops.gml_mobilenet_series import GMLMBBlock, GMLSELayer


def _range_to_list(range_: List[int]) -> List[int]:
    assert len(range_) == 3

    start, end, step = range_
    return list(range(start, end + 1, step))


def _mutate_conv_module(
    conv_module: ConvModule,
    derived_in_channels: Optional[BaseMutable] = None,
    derived_out_channels: Optional[BaseMutable] = None,
    kernel_size_derive_cfg: Optional[Tuple[BaseMutable,
                                           Sequence[int]]] = None):
    if derived_in_channels is not None:
        conv_module.conv.mutate_in_channels(derived_in_channels)
    if derived_out_channels is not None:
        conv_module.conv.mutate_out_channels(derived_out_channels)
        if conv_module.with_norm:
            conv_module.bn.mutate_num_features(derived_out_channels)

    if kernel_size_derive_cfg is not None:
        mutable_kernel_size, kernel_size_list = kernel_size_derive_cfg
        conv_module.conv.mutate_kernel_size(
            mutable_kernel_size.derive_same_mutable(), kernel_size_list)


def _mutate_se_layer(se_layer: GMLSELayer, in_channels_mutable: BaseMutable,
                     divisor: int):
    se_layer.conv1.conv.mutate_in_channels(
        in_channels_mutable.derive_same_mutable())
    se_layer.conv1.conv.mutate_out_channels(
        in_channels_mutable.derive_divide_mutable(divisor))
    if se_layer.conv1.with_norm:
        se_layer.conv1.bn.mutate_num_features(
            in_channels_mutable.derive_divide_mutable(divisor))

    se_layer.conv2.conv.mutate_in_channels(
        in_channels_mutable.derive_divide_mutable(divisor))
    se_layer.conv2.conv.mutate_out_channels(
        in_channels_mutable.derive_same_mutable())
    if se_layer.conv2.with_norm:
        se_layer.conv2.bn.mutate_num_features(
            in_channels_mutable.derive_same_mutable())


@MODELS.register_module()
class AttentiveMobileNet(BaseBackbone):
    # Parameters to build layers. 6 parameters are needed to construct a
    # layer, from left to right:
    # [min_expand_ratio, max_expand_ratio, step]
    # [min_channel, max_channel, step]
    # [min_num_blocks, max_num_blocks, step]
    # [min_kernel_size, max_kernel_size, step]
    # stride
    # se_cfg
    arch_settings = [
        [[1, 1, 1], [16, 24, 8], [1, 2, 1], [3, 5, 2], 1, False],
        [[4, 6, 1], [24, 32, 8], [3, 5, 1], [3, 5, 2], 2, False],
        [[4, 6, 1], [32, 40, 8], [3, 6, 1], [3, 5, 2], 2, True],
        [[4, 6, 1], [64, 72, 8], [3, 6, 1], [3, 5, 2], 2, False],
        [[4, 6, 1], [112, 128, 8], [3, 8, 1], [3, 5, 2], 1, True],
        [[6, 6, 1], [192, 216, 8], [3, 8, 1], [3, 5, 2], 2, True],
        [[6, 6, 1], [216, 224, 8], [1, 2, 1], [3, 5, 2], 1, True],
    ]

    def __init__(self,
                 first_out_channels_range=[16, 24, 8],
                 last_out_channels_range=[1792, 1984, 1984 - 1792],
                 last_expand_ratio=6,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=dict(type='CenterCropDynamicConv2d'),
                 norm_cfg=dict(type='DynamicBatchNorm2d'),
                 act_cfg=dict(type='ReLU'),
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

        self.first_conv = ConvModule(
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
        self.source_mutables['first_conv'] = mutable_out_channels
        _mutate_conv_module(
            self.first_conv, derived_out_channels=mutable_out_channels * 1)
        self.last_mutable = mutable_out_channels

        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            print(f'stage {i + 1}: {layer_cfg}')
            expand_ratio_range, channel_range, num_blocks_range, \
                kernel_size_range, stride, use_se = layer_cfg
            out_channels_list = [
                make_divisible(x * widen_factor, 8)
                for x in _range_to_list(channel_range)
            ]
            num_blocks_list = _range_to_list(num_blocks_range)
            kernel_size_list = _range_to_list(kernel_size_range)
            expand_ratio_list = _range_to_list(expand_ratio_range)

            inverted_res_layer = self.make_layer(
                out_channels_list=out_channels_list,
                num_blocks_list=num_blocks_list,
                kernel_size_list=kernel_size_list,
                stride=stride,
                expand_ratio_list=expand_ratio_list,
                use_se=use_se,
                stage_idx=i + 1)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        last_out_channels = [
            make_divisible(x * widen_factor, 8)
            for x in _range_to_list(last_out_channels_range)
        ]
        out_channels = max(last_out_channels)

        last_mutable = self.last_mutable
        # align with gml
        last_layers = OrderedDict([
            ('final_expand_layer',
             ConvModule(
                 in_channels=self.in_channels,
                 out_channels=self.in_channels * last_expand_ratio,
                 kernel_size=1,
                 padding=0,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg)), ('pool', nn.AdaptiveAvgPool2d(
                     (1, 1))),
            ('feature_mix_layer',
             ConvModule(
                 in_channels=self.in_channels * last_expand_ratio,
                 out_channels=out_channels,
                 kernel_size=1,
                 padding=0,
                 bias=False,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=None,
                 act_cfg=self.act_cfg))
        ])
        _mutate_conv_module(
            last_layers['final_expand_layer'],
            derived_in_channels=last_mutable * 1,
            derived_out_channels=last_mutable * last_expand_ratio)
        mutable_out_channels = OneShotMutableChannel(
            num_channels=out_channels,
            candidate_choices=last_out_channels,
            candidate_mode='number')
        self.source_mutables['last_conv'] = mutable_out_channels
        self.last_mutable = mutable_out_channels
        _mutate_conv_module(
            last_layers['feature_mix_layer'],
            derived_in_channels=last_mutable * last_expand_ratio,
            derived_out_channels=mutable_out_channels * 1)

        self.add_module('last_conv', Sequential(last_layers))
        self.layers.append('last_conv')

    def make_layer(self, out_channels_list, num_blocks_list, kernel_size_list,
                   stride, expand_ratio_list, use_se, stage_idx):
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
        expand_ratio = max(expand_ratio_list)
        if len(expand_ratio_list) > 1:
            mutable_expand_value = OneShotMutableValue(
                value_list=expand_ratio_list, default_value=expand_ratio)
            mutable_list.append(mutable_expand_value)
        else:
            mutable_expand_value = None
        expand_fn = lambda x: x[0] * x[1]  # noqa: E731
        mutable_out_channels = OneShotMutableChannel(
            num_channels=out_channels,
            candidate_choices=out_channels_list,
            candidate_mode='number')
        passed_out_channels = False

        layers = []
        for i in range(max(num_blocks_list)):
            if i >= 1:
                stride = 1
            if use_se:
                se_cfg = dict(
                    act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')),
                    ratio=4,
                    conv_cfg=self.conv_cfg,
                    use_avgpool=False)
            else:
                se_cfg = None
            mb_layer = GMLMBBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp,
                se_cfg=se_cfg,
                with_attentive_shortcut=True)

            if mutable_expand_value is not None:
                channels_derive_cfg = (last_mutable, mutable_expand_value)
            else:
                channels_derive_cfg = (last_mutable, expand_ratio)

            if mb_layer.with_expand_conv:
                _mutate_conv_module(
                    mb_layer.expand_conv,
                    derived_in_channels=last_mutable * 1,
                    derived_out_channels=expand_fn(channels_derive_cfg))

            _mutate_conv_module(
                mb_layer.depthwise_conv,
                derived_in_channels=expand_fn(channels_derive_cfg),
                derived_out_channels=expand_fn(channels_derive_cfg),
                kernel_size_derive_cfg=(mutable_kernel_size, kernel_size_list))

            if mb_layer.with_se:
                _mutate_se_layer(
                    mb_layer.se,
                    in_channels_mutable=mb_layer.depthwise_conv.conv.
                    mutable_out_channels,
                    divisor=4)

            if not mb_layer.with_res_shortcut:
                if mb_layer.with_attentive_shortcut:
                    mb_layer.shortcut.conv.mutate_in_channels(last_mutable * 1)
                    mb_layer.shortcut.conv.mutate_out_channels(
                        mutable_out_channels * 1)

                if not passed_out_channels:
                    last_mutable = mutable_out_channels
                    mutable_list.append(last_mutable)
                    passed_out_channels = True

            _mutate_conv_module(
                mb_layer.linear_conv,
                derived_in_channels=expand_fn(channels_derive_cfg),
                derived_out_channels=last_mutable * 1)

            layers.append(mb_layer)
            self.in_channels = out_channels

        dynamic_seq = DynamicSequential(*layers)
        mutable_depth = OneShotMutableValue(
            value_list=num_blocks_list, default_value=max(num_blocks_list))
        mutable_list.append(mutable_depth)
        dynamic_seq.mutate_depth(mutable_depth, depth_seq=num_blocks_list)

        self.source_mutables[f'stage_{stage_idx}'] = mutable_list
        self.last_mutable = last_mutable

        return dynamic_seq

    def forward(self, x):
        x = self.first_conv(x)

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
