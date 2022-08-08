# Copyright (c) OpenMMLab. All rights reserved.
from math import log
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from ...ops.darts_series import DartsSepConv
from .base_connector import BaseConnector


@MODELS.register_module()
class BYOTConncetor(BaseConnector):
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
        num_classes: int,
        expansion: int = 4,
        pool_size: Union[int, Tuple[int]] = 4,
        kernel_size: Union[int, Tuple[int]] = 1,
        stride: Union[int, Tuple[int]] = 1,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.attention = nn.Sequential(
            DartsSepConv(
                C_in=in_channel * expansion,
                C_out=in_channel * expansion,
                kernel_size=kernel_size,
                stride=stride), nn.BatchNorm2d(in_channel * expansion),
            nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid())
        scala_num = log(out_channel / in_channel, 2)
        assert scala_num.is_integer()
        scala = []

        _in_channel = in_channel

        for _ in range(int(scala_num)):
            scala.append(
                DartsSepConv(
                    C_in=_in_channel * expansion,
                    C_out=_in_channel * 2 * expansion,
                    kernel_size=kernel_size,
                    stride=stride))
            _in_channel *= 2
        scala.append(nn.AvgPool2d(*pool_size))
        self.scala = nn.Sequential(*scala)
        self.fc = nn.Linear(out_channel * expansion, num_classes)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        feat = self.attention(feature)
        feat = feat * feature

        feat = self.scala(feat)
        feat = feat.view(feature.size(0), -1)
        logits = self.fc(feat)
        return feat, logits
