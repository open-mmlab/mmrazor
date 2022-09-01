# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch.nn as nn
from mmcv.cnn import ConvModule

from mmrazor.registry import MODELS
from .base import BaseOP

try:
    from mmcls.models.utils import SELayer
except ImportError:
    from mmrazor.utils import get_placeholder
    SELayer = get_placeholder('mmcls')


@MODELS.register_module()
class ConvBnAct(BaseOP):
    """ConvBnAct block from timm.

    Args:
        in_channels (int): number of in channels.
        out_channels (int): number of out channels.
        kernel_size (int): kernel size of convolution.
        stride (int, optional): stride of convolution. Defaults to 1.
        dilation (int, optional): dilation rate of convolution. Defaults to 1.
        padding (int, optional): padding size of convolution. Defaults to 0.
        skip (bool, optional): whether using skip connect. Defaults to False.
        conv_cfg (Optional[dict], optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (Dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (Dict, optional):Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 padding: int = 0,
                 skip: bool = False,
                 conv_cfg: Optional[dict] = None,
                 se_cfg: Dict = None,
                 norm_cfg: Dict = dict(type='BN'),
                 act_cfg: Dict = dict(type='ReLU')):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.has_residual = skip and stride == 1 \
            and in_channels == out_channels
        self.with_se = se_cfg is not None

        if self.with_se:
            assert isinstance(se_cfg, dict)
            self.se = SELayer(self.out_channels, **se_cfg)

        self.convModule = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        shortcut = x
        x = self.convModule(x)
        if self.has_residual:
            x += shortcut
        return x


@MODELS.register_module()
class DepthwiseSeparableConv(BaseOP):
    """DepthwiseSeparable block Used for DS convs in MobileNet-V1 and in the
    place of IR blocks that have no expansion (factor of 1.0). This is an
    alternative to having a IR with an optional first pw conv.

    Args:
        in_channels (int): number of in channels.
        out_channels (int): number of out channels.
        dw_kernel_size (int, optional): the kernel size of depth-wise
            convolution. Defaults to 3.
        stride (int, optional): stride of convolution.
            Defaults to 1.
        dilation (int, optional): dilation rate of convolution.
            Defaults to 1.
        noskip (bool, optional): whether use skip connection.
            Defaults to False.
        pw_kernel_size (int, optional): kernel size of point wise convolution.
            Defaults to 1.
        pw_act (bool, optional): whether using activation in point-wise
            convolution. Defaults to False.
        se_cfg (Dict, optional): _description_. Defaults to None.
        conv_cfg (Optional[dict], optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (Dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (Dict, optional):Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dw_kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 noskip: bool = False,
                 pw_kernel_size: int = 1,
                 pw_act: bool = False,
                 conv_cfg: Optional[dict] = None,
                 se_cfg: Dict = None,
                 norm_cfg: Dict = dict(type='BN'),
                 act_cfg: Dict = dict(type='ReLU')):

        super().__init__(
            in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.has_residual = (stride == 1
                             and in_channels == out_channels) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.se_cfg = se_cfg

        self.conv_dw = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=dw_kernel_size // 2,
            groups=in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Squeeze-and-excitation
        self.se = SELayer(out_channels, **
                          se_cfg) if self.se_cfg else nn.Identity()

        self.conv_pw = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=pw_kernel_size,
            padding=pw_kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg if self.has_pw_act else None,
        )

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.se(x)
        x = self.conv_pw(x)
        if self.has_residual:
            x += shortcut
        return x
