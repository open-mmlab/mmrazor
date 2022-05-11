# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule

from mmrazor.registry import MODELS
from .base import BaseOP


@MODELS.register_module()
class Identity(BaseOP):
    """Base class for searchable operations.

    Args:
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: None.
    """

    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=None,
                 **kwargs):
        super(Identity, self).__init__(**kwargs)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x
