# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath

from mmrazor.registry import MODELS
from .base import BaseOP


@MODELS.register_module()
class DartsPoolBN(BaseOP):

    def __init__(self,
                 pool_type,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 use_drop_path=False,
                 **kwargs):
        super(DartsPoolBN, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(self.kernel_size, self.stride, 1)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(
                self.kernel_size, self.stride, 1, count_include_pad=False)
        self.bn = build_norm_layer(self.norm_cfg, self.out_channels)[1]

        self.drop_path = DropPath() if use_drop_path else None

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        if self.drop_path is not None:
            out = self.drop_path(out)

        return out


@MODELS.register_module()
class DartsDilConv(BaseOP):

    def __init__(self,
                 kernel_size,
                 use_drop_path=False,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super(DartsDilConv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.dilation = 2
        assert self.kernel_size in [3, 5]
        assert self.stride in [1, 2]
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                self.kernel_size,
                self.stride, (self.kernel_size // 2) * self.dilation,
                dilation=self.dilation,
                groups=self.in_channels,
                bias=False),
            nn.Conv2d(
                self.in_channels, self.out_channels, 1, stride=1, bias=False),
            build_norm_layer(self.norm_cfg, self.in_channels)[1])

        self.drop_path = DropPath() if use_drop_path else None

    def forward(self, x):
        out = self.conv1(x)
        if self.drop_path is not None:
            out = self.drop_path(out)
        return out


@MODELS.register_module()
class DartsSepConv(BaseOP):

    def __init__(self,
                 kernel_size,
                 use_drop_path=False,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super(DartsSepConv, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        assert self.kernel_size in [3, 5]
        assert self.stride in [1, 2]
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                self.kernel_size,
                self.stride,
                self.kernel_size // 2,
                groups=self.in_channels,
                bias=False),
            nn.Conv2d(
                self.in_channels, self.in_channels, 1, stride=1, bias=False),
            build_norm_layer(self.norm_cfg, self.in_channels)[1])
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                1,
                self.kernel_size // 2,
                groups=self.in_channels,
                bias=False),
            nn.Conv2d(
                self.out_channels, self.out_channels, 1, stride=1, bias=False),
            build_norm_layer(self.norm_cfg, self.out_channels)[1])

        self.drop_path = DropPath() if use_drop_path else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.drop_path is not None:
            out = self.drop_path(out)
        return out


@MODELS.register_module()
class DartsSkipConnect(BaseOP):
    """Reduce feature map size by factorized pointwise (stride=2)."""

    def __init__(self,
                 use_drop_path=False,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super(DartsSkipConnect, self).__init__(**kwargs)
        self.norm_cfg = norm_cfg
        if self.stride > 1:
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(
                self.in_channels,
                self.out_channels // 2,
                1,
                stride=2,
                padding=0,
                bias=False)
            self.conv2 = nn.Conv2d(
                self.in_channels,
                self.out_channels // 2,
                1,
                stride=2,
                padding=0,
                bias=False)
            self.bn = build_norm_layer(self.norm_cfg, self.out_channels)[1]

        self.drop_path = DropPath() if use_drop_path else None

    def forward(self, x):
        if self.stride > 1:
            x = self.relu(x)
            out = torch.cat(
                [self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
            out = self.bn(out)
            if self.drop_path is not None:
                out = self.drop_path(out)
        else:
            out = x
        return out


@MODELS.register_module()
class DartsZero(BaseOP):

    def __init__(self, **kwargs):
        super(DartsZero, self).__init__(**kwargs)

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
