# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmrazor.registry import MODELS
from .base import BaseOP

try:
    from mmcls.models.utils import channel_shuffle
except ImportError:
    from mmrazor.utils import get_placeholder
    channel_shuffle = get_placeholder('mmcls')


@MODELS.register_module()
class ShuffleBlock(BaseOP):
    """InvertedResidual block for Searchable ShuffleNetV2 backbone.

    Args:
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution layer. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 kernel_size,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 **kwargs):

        super(ShuffleBlock, self).__init__(**kwargs)

        assert kernel_size in [3, 5, 7]
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp

        branch_features = self.out_channels // 2
        if self.stride == 1:
            assert self.in_channels == branch_features * 2, (
                f'in_channels ({self.in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if self.in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                    groups=self.in_channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None),
                ConvModule(
                    self.in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                self.in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                groups=branch_features,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@MODELS.register_module()
class ShuffleXception(BaseOP):
    """Xception block for ShuffleNetV2 backbone.

    Args:
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 **kwargs):
        super(ShuffleXception, self).__init__(**kwargs)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.mid_channels = self.out_channels // 2

        branch_features = self.out_channels // 2
        if self.stride == 1:
            assert self.in_channels == branch_features * 2, (
                f'in_channels ({self.in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if self.in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=self.in_channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None),
                ConvModule(
                    self.in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
            )

        self.branch2 = []

        self.branch2.append(
            DepthwiseSeparableConvModule(
                self.in_channels if (self.stride > 1) else branch_features,
                self.mid_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg), )
        self.branch2.append(
            DepthwiseSeparableConvModule(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg))
        self.branch2.append(
            DepthwiseSeparableConvModule(
                self.mid_channels,
                branch_features,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg))
        self.branch2 = nn.Sequential(*self.branch2)

    def forward(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out
