# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class SingleConvConnector(BaseConnector):
    """General connector which only contains a conv layer.

    Args:
        in_channel (int): The input channel of the connector.
        out_channel (int): The output channel of the connector.
        conv_cfg (dict, optional): The config to control the convolution.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        conv_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.conv = build_conv_layer(
            conv_cfg, in_channel, out_channel, kernel_size=1, stride=1)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.conv(feature)

    def init_weights(self) -> None:
        """Init parameters.

        In the subclass of ``BaseModule``, `init_weights` will be called
        automativally.
        """
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    device = m.weight.device
                    in_channels, _, k1, k2 = m.weight.shape
                    m.weight[:] = torch.randn(
                        m.weight.shape, device=device) / np.sqrt(
                            k1 * k2 * in_channels) * 1e-4
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    continue


@MODELS.register_module()
class ConvBNConnector(BaseConnector):
    """General connector which contains a conv layer with BN.

    Args:
        in_channel (int): The input channels of the connector.
        out_channel (int): The output channels of the connector.
        conv_cfg (dict, optional): The config to control the convolution.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        conv_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channel,
            out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        _, self.bn = build_norm_layer(dict(type='BN'), out_channel)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.bn(self.conv(feature))


@MODELS.register_module()
class ConvBNReLUConnector(BaseConnector):
    """General connector which contains a conv layer with BN and ReLU.

    Args:
        in_channel (int): The input channels of the connector.
        out_channel (int): The output channels of the connector.
        conv_cfg (dict, optional): The config to control the convolution.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        conv_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.conv = build_conv_layer(
            conv_cfg, in_channel, out_channel, kernel_size=1)
        _, self.bn = build_norm_layer(dict(type='BN'), out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.relu(self.bn(self.conv(feature)))
