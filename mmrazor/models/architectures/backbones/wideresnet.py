# Copyright (c) OpenMMLab. All rights reserved.
# This file is modified from `mmcls.models.backbones.resnet`

import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init

from mmrazor.registry import MODELS


class BasicBlock(nn.Module):
    """BasicBlock for WideResNet. The differences from ResNet are in:
     1. The forward path
     2. The position of residual path
     3. Different downsample

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        droprate (float, optional): droprate of the block. Defaults to 0.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 1,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module = None,
        droprate: float = 0,
        conv_cfg: Dict = None,
        norm_cfg: Dict = dict(type='BN')
    ) -> None:  # noqa: E125
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.stride = stride
        self.dilation = dilation
        self.droprate = droprate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward func.

        Args:
            x (torch.Tensor): input.

        Returns:
            torch.Tensor: output.
        """

        identity = self.relu1(self.bn1(x))
        out = self.conv1(identity)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if self.downsample:
            out += self.downsample(identity)
        else:
            out += x
        return out


def get_expansion(block: nn.Module,
                  widen_factor: int,
                  expansion: int = None) -> int:
    """Get the expansion of a residual block.
    The block expansion will be obtained by the following order:
    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. If ``block`` is ``BaseBlock``, then return ``widen_factor``.
    3. Return the default value according the the block type:
       4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        widen_factor (int): The given widen factor.
        expansion (int | None): The given expansion ratio.
    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = widen_factor
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck.
        droprate (float, optional): droprate of the layer. Defaults to 0.
        stride (int): stride of the first block. Default: 1.
        conv_cfg (Dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (Dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block: nn.Module,
                 num_blocks: int,
                 in_channels: int,
                 out_channels: int,
                 expansion: int,
                 droprate: float = 0,
                 stride: int = 1,
                 conv_cfg: Dict = None,
                 norm_cfg: Dict = dict(type='BN'),
                 **kwargs):
        self.block = block
        self.droprate = droprate
        self.expansion = expansion

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@MODELS.register_module()
class WideResNet(BaseModule):
    """WideResNet backbone. Only support 3-stage WideResNet, which is usually
    for tiny images. E.g., CIFAR10 and CIFAR100.

    WRN50 and WRN101 are now officially supported in
    MMClassification. See link below:
    https://github.com/open-mmlab/mmclassification/pull/715

    Please refer to the `paper <https://arxiv.org/abs/1605.07146>`__ for
    details.

    Args:
        depth (int): Network depth, from {10, 16, 22, 28, 40, 50, 101, 152}.
        widen_factor (int):  Width multiplier of mid-channel in blocks.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """
    arch_setting = {
        10: (BasicBlock, (1, 1, 1)),
        16: (BasicBlock, (2, 2, 2)),
        22: (BasicBlock, (3, 3, 3)),
        28: (BasicBlock, (4, 4, 4)),
        40: (BasicBlock, (6, 6, 6)),
    }

    def __init__(self,
                 depth: int,
                 widen_factor: int = 4,
                 in_channels: int = 3,
                 stem_channels: int = 16,
                 base_channels: int = 16,
                 expansion: int = None,
                 num_stages: int = 3,
                 strides: Tuple[int, ...] = (1, 2, 2),
                 dilations: Tuple[int, ...] = (1, 1, 1),
                 frozen_stages: int = -1,
                 conv_cfg: Dict = None,
                 norm_cfg: Dict = dict(type='BN', requires_grad=True),
                 norm_eval: bool = False,
                 zero_init_residual: bool = False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(WideResNet, self).__init__(init_cfg)
        if depth > 40:
            """MMClassication now supports WRN-50 and 101 officially.

            Refer to:
            https://github.com/open-mmlab/mmclassification/pull/715/files
            """
            warnings.warn('`WiderResNet` deep than 40 now is deprecated')
        if depth not in self.arch_setting:
            raise KeyError(f'invalid depth {depth} for WideResNet')
        self.depth = depth
        self.widen_factor = widen_factor
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_setting[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, widen_factor, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            )
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, _out_channels // 2, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(WideResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        # TODO: return multi-stage features.
        x = self.conv1(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.norm1(x)
        x = self.relu(x)
        return tuple([x])
