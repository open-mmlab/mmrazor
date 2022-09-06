# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model import Sequential, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_ops.bricks import DynamicSequential
from mmrazor.models.architectures.ops.gml_mobilenet_series import (GMLMBBlock,
                                                                   GMLSELayer)
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS

logger = MMLogger.get_current_instance()


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
        conv_module.conv.register_mutable_attr('in_channels',
                                               mutable_in_channels)
    if mutable_out_channels is not None:
        conv_module.conv.register_mutable_attr('out_channels',
                                               mutable_out_channels)
        if conv_module.with_norm:
            conv_module.bn.register_mutable_attr('num_features',
                                                 mutable_out_channels)
    if mutable_kernel_size is not None:
        conv_module.conv.register_mutable_attr('kernel_size',
                                               mutable_kernel_size)


def _mutate_se_layer(se_layer: GMLSELayer, mutable_in_channels: BaseMutable,
                     se_cfg: dict):
    # TODO: make divisiable
    ratio = se_cfg.get('ratio', 16)
    divisor = se_cfg.get('divisor', 8)
    derived_mid_channels = mutable_in_channels.derive_divide_mutable(
        ratio, divisor)
    _mutate_conv_module(
        se_layer.conv1,
        mutable_in_channels=mutable_in_channels,
        mutable_out_channels=derived_mid_channels)
    _mutate_conv_module(
        se_layer.conv2,
        mutable_in_channels=derived_mid_channels,
        mutable_out_channels=mutable_in_channels)


def _mutate_mb_layer(mb_layer: GMLMBBlock, mutable_in_channels,
                     mutable_out_channels, mutable_expand_value,
                     mutable_kernel_size, se_cfg):
    # mutate in_channels, out_channels, kernel_size for mb_layer
    derived_expand_channels = mutable_in_channels * mutable_expand_value

    if mb_layer.with_expand_conv:
        _mutate_conv_module(
            mb_layer.expand_conv,
            mutable_in_channels=mutable_in_channels,
            mutable_out_channels=derived_expand_channels)

    _mutate_conv_module(
        mb_layer.depthwise_conv,
        mutable_in_channels=derived_expand_channels,
        mutable_out_channels=derived_expand_channels,
        mutable_kernel_size=mutable_kernel_size)

    if mb_layer.with_se:
        _mutate_se_layer(
            mb_layer.se,
            mutable_in_channels=derived_expand_channels,
            se_cfg=se_cfg)

    if not mb_layer.with_res_shortcut:
        if mb_layer.with_attentive_shortcut:
            mb_layer.shortcut.conv.register_mutable_attr(
                'in_channels', mutable_in_channels)
            mb_layer.shortcut.conv.register_mutable_attr(
                'out_channels', mutable_out_channels)

    _mutate_conv_module(
        mb_layer.linear_conv,
        mutable_in_channels=derived_expand_channels,
        mutable_out_channels=mutable_out_channels)


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
                 dropout_stages=6,
                 conv_cfg=dict(type='BigNasConv2d'),
                 norm_cfg=dict(type='DynamicBatchNorm2d'),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 zero_init_residual=True,
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
        if dropout_stages not in range(-1, 8):
            raise ValueError('dropout_stages must be in range(-1, 8). '
                             f'But received {dropout_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dropout_stages = dropout_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.with_cp = with_cp

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
        _mutate_conv_module(
            self.first_conv, mutable_out_channels=mutable_out_channels)
        self.last_mutable = mutable_out_channels

        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
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
        derived_expand_channels = self.last_mutable * last_expand_ratio
        mutable_out_channels = OneShotMutableChannel(
            num_channels=out_channels,
            candidate_choices=last_out_channels,
            candidate_mode='number')
        _mutate_conv_module(
            last_layers['final_expand_layer'],
            mutable_in_channels=self.last_mutable,
            mutable_out_channels=derived_expand_channels)
        _mutate_conv_module(
            last_layers['feature_mix_layer'],
            mutable_in_channels=derived_expand_channels,
            mutable_out_channels=mutable_out_channels)

        self.last_mutable = mutable_out_channels
        self.add_module('last_conv', Sequential(last_layers))
        self.layers.append('last_conv')
        self.blocks = self.layers[:-1]

        logger.info(f'layers:\n {self.layers}')
        logger.info(f'blocks:\n {self.blocks}')

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
        out_channels = max(out_channels_list)
        kernel_size = max(kernel_size_list)
        expand_ratio = max(expand_ratio_list)

        mutable_out_channels = OneShotMutableChannel(
            num_channels=out_channels,
            candidate_choices=out_channels_list,
            candidate_mode='number')
        mutable_kernel_size = OneShotMutableValue(
            value_list=kernel_size_list, default_value=kernel_size)
        mutable_expand_value = OneShotMutableValue(
            value_list=expand_ratio_list, default_value=expand_ratio)

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

            _mutate_mb_layer(mb_layer, self.last_mutable, mutable_out_channels,
                             mutable_expand_value, mutable_kernel_size, se_cfg)
            self.last_mutable = mutable_out_channels
            layers.append(mb_layer)
            self.in_channels = out_channels

        dynamic_seq = DynamicSequential(*layers)
        mutable_depth = OneShotMutableValue(
            value_list=num_blocks_list, default_value=max(num_blocks_list))
        dynamic_seq.register_mutable_attr('depth', mutable_depth)

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

    def set_dropout(self, drop_prob: float) -> None:
        total_block_nums = len(self.blocks)
        visited_block_nums = 0
        for idx, layer_name in enumerate(self.blocks, start=1):
            layer = getattr(self, layer_name)
            assert isinstance(layer, DynamicSequential)
            visited_block_nums += len(layer)
            if idx < self.dropout_stages:
                continue

            for mb_idx, mb_layer in enumerate(layer):
                if isinstance(mb_layer, GMLMBBlock):
                    ratio = (visited_block_nums - len(layer) +
                             mb_idx) / total_block_nums
                    mb_drop_prob = drop_prob * ratio
                    mb_layer.drop_prob = mb_drop_prob

                    logger.debug(f'set drop prob `{mb_drop_prob}` '
                                 f'to layer: {layer_name}.{mb_idx}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.first_conv.parameters():
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

    def init_weights(self) -> None:
        super().init_weights()

        if self.zero_init_residual:
            for name, module in self.named_modules():
                if isinstance(module, GMLMBBlock):
                    if module.with_res_shortcut or \
                            module.with_attentive_shortcut:
                        norm_layer = module.linear_conv.norm
                        constant_init(norm_layer, val=0)
                        logger.debug(
                            f'init {type(norm_layer)} of linear_conv in '
                            f'`{name}` to zero')
