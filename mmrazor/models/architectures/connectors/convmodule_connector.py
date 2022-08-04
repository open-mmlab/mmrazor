# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, build_padding_layer)
from mmcv.utils import _BatchNorm, _InstanceNorm

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class ConvModuleConncetor(BaseConnector):
    """Convolution connector that bundles conv/norm/activation layers.

    Args:
        in_channel (int): The input channel of the connector.
        out_channel (int): The output channel of the connector.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Tuple[int]] = 1,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: Union[str, bool] = 'auto',
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Dict = dict(type='ReLU'),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = 'zeros',
        order: tuple = ('conv', 'norm', 'act'),
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        assert conv_cfg is None or isinstance(conv_cfg, dict), \
            'conv_cfg must be None or a dict, ' \
            f'but got {type(conv_cfg).__name__}.'
        assert norm_cfg is None or isinstance(norm_cfg, dict), \
            'norm_cfg must be None or a dict, ' \
            f'but got {type(norm_cfg).__name__}.'
        assert act_cfg is None or isinstance(act_cfg, dict), \
            'act_cfg must be None or a dict, ' \
            f'but got {type(act_cfg).__name__}.'

        official_padding_mode = ['zeros', 'circular']
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3, \
            '"order" must be a tuple and with length 3.'
        assert set(order) == set(['conv', 'norm', 'act']), \
            'the set of "order" must be equal to the set of ' \
            '["conv", "norm", "act"].'

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channel
            else:
                norm_channels = in_channel
            self.norm_name, self.norm = build_norm_layer(
                norm_cfg, norm_channels)
            self.add_module(self.norm_name, self.norm)
            if self.with_bias:
                if isinstance(self.norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    feature = self.padding_layer(feature)
                feature = self.conv(feature)
            elif layer == 'norm' and self.with_norm:
                feature = self.norm(feature)
            elif layer == 'act' and self.with_activation:
                feature = self.activate(feature)
        return feature
