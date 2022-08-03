# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath, build_conv_layer
from mmengine import is_tuple_of
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmrazor.models.architectures.dynamic_op import DynamicConv2d
from mmrazor.registry import MODELS
from .base import BaseOP


class GMLSELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        squeeze_channels (None or int): The intermediate channel number of
            SElayer. Default: None, means the value of ``squeeze_channels``
            is ``make_divisible(channels // ratio, divisor)``.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will
            be ``make_divisible(channels // ratio, divisor)``. Only used when
            ``squeeze_channels`` is None. Default: 16.
        divisor(int): The divisor to true divide the channel number. Only
            used when ``squeeze_channels`` is None. Default: 8.
        conv_cfg (None or dict): Config dict for convolution layer. Default:
            None, which means using conv2d.
        return_weight(bool): Whether to return the weight. Default: False.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 squeeze_channels=None,
                 ratio=16,
                 divisor=8,
                 bias='auto',
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 return_weight=False,
                 use_avgpool=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) if use_avgpool else None
        if squeeze_channels is None:
            squeeze_channels = make_divisible(channels // ratio, divisor)
        assert isinstance(squeeze_channels, int) and squeeze_channels > 0, \
            '"squeeze_channels" should be a positive integer, but get ' + \
            f'{squeeze_channels} instead.'
        self.return_weight = return_weight
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=squeeze_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        if self.global_avgpool is not None:
            out = self.global_avgpool(x)
        else:
            out = x.mean(3, keepdim=True).mean(2, keepdim=True)

        out = self.conv1(out)
        out = self.conv2(out)
        if self.return_weight:
            return out
        else:
            return x * out


class ShortcutLayer(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 reduction: int = 1,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 init_cfg=None):
        super().__init__(init_cfg)

        assert reduction in [1, 2]
        self.reduction = reduction

        # TODO
        # to align with gml
        self.with_conv = in_channels != out_channels
        # conv module can be removed if in_channels equal to out_channels
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.reduction > 1:
            padding = x.size(-1) & 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)

        # HACK
        if isinstance(self.conv, DynamicConv2d):
            mutable_in_channels = self.conv.mutable_in_channels
            mutable_out_channels = self.conv.mutable_out_channels
            if mutable_out_channels is not None and \
                    mutable_in_channels is not None:
                if mutable_out_channels.current_mask.sum().item() != \
                        mutable_in_channels.current_mask.sum().item():
                    x = self.conv(x)
        else:
            if self.with_conv:
                x = self.conv(x)

        return x


@MODELS.register_module()
class GMLMBBlock(BaseOP):
    """Mobilenet block for Searchable backbone.

    Args:
        kernel_size (int): Size of the convolving kernel.
        expand_ratio (int): The input channels' expand factor of the depthwise
             convolution.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 kernel_size,
                 expand_ratio,
                 se_cfg=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 with_attentive_shortcut=False,
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
            and not with_attentive_shortcut)
        assert self.stride in [1, 2]
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
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
            self.se = GMLSELayer(self.mid_channels, **se_cfg)
        self.linear_conv = ConvModule(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

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
                return x + self.drop_path(out)
            elif self.with_attentive_shortcut:
                return self.shortcut(x) + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out
