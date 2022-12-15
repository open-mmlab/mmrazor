# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_conv_layer
from mmcv.cnn.bricks.drop import drop_path

from mmrazor.registry import MODELS
from .base import BaseOP

try:
    from mmcls.models.utils import SELayer
except ImportError:
    from mmrazor.utils import get_placeholder
    SELayer = get_placeholder('mmcls')


class ShortcutLayer(BaseOP):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 reduction: int = 1,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)

        assert reduction in [1, 2]
        self.reduction = reduction

        # conv module can be removed if in_channels equal to out_channels
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction > 1:
            padding = x.size(-1) & 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)

        # HACK
        if hasattr(self.conv, 'mutable_in_channels'
                   ) and self.conv.mutable_in_channels is not None:
            in_channels = self.conv.mutable_in_channels.current_mask.sum(
            ).item()
        else:
            in_channels = self.conv.in_channels
        if hasattr(self.conv, 'mutable_out_channels'
                   ) and self.conv.mutable_out_channels is not None:
            out_channels = self.conv.mutable_out_channels.current_mask.sum(
            ).item()
        else:
            out_channels = self.conv.out_channels

        if in_channels != out_channels:
            x = self.conv(x)

        return x


@MODELS.register_module()
class MBBlock(BaseOP):
    """Mobilenet block for Searchable backbone.

    Args:
        kernel_size (int): Size of the convolving kernel.
        expand_ratio (int): The input channels' expand factor of the depthwise
             convolution.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        with_attentive_shortcut (bool): Use shortcut in AttentiveNAS or not.
            Defaults to False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 kernel_size: int,
                 expand_ratio: int,
                 se_cfg: Dict = None,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 norm_cfg: Dict = dict(type='BN'),
                 act_cfg: Dict = dict(type='ReLU'),
                 drop_path_rate: float = 0.,
                 with_cp: bool = False,
                 with_attentive_shortcut: bool = False,
                 **kwargs):

        super().__init__(**kwargs)

        if with_attentive_shortcut:
            self.shortcut = ShortcutLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                reduction=self.stride,
                conv_cfg=conv_cfg)
        self.with_attentive_shortcut = with_attentive_shortcut

        self.with_res_shortcut = (
            self.stride == 1 and self.in_channels == self.out_channels
            and not self.with_attentive_shortcut)
        assert self.stride in [1, 2]
        self._drop_path_rate = drop_path_rate
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.mid_channels = self.in_channels * expand_ratio
        self.with_expand_conv = (self.mid_channels != self.in_channels)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=kernel_size // 2,
            groups=self.mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(self.mid_channels, **se_cfg)
        self.linear_conv = ConvModule(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    @property
    def drop_path_rate(self):
        return self._drop_path_rate

    @drop_path_rate.setter
    def drop_path_rate(self, value):
        if not isinstance(value, float):
            raise TypeError('Expected float.')
        self._drop_path_rate = value

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """

        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)
            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                if self.drop_path_rate > 0.:
                    out = drop_path(out, self.drop_path_rate, self.training)
                return x + out

            elif self.with_attentive_shortcut:
                sx = self.shortcut(x)
                if self.drop_path_rate > 0. and \
                        x.size(1) == sx.size(1) and \
                        self.shortcut.reduction == 1:
                    out = drop_path(out, self.drop_path_rate, self.training)
                return sx + out

            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out
