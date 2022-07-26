# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class SingleConvConnector(BaseConnector):
    """General connector which only contains a conv layer.

    Args:
        in_channel (int): The input channel of the connector.
        out_channel (int): The output channel of the connector.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.init_parameters()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.conv(feature)

    def init_parameters(self) -> None:
        """Init parameters."""
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
class BNConnector(BaseConnector):
    """General connector which contains a conv layer with BN.

    Args:
        in_channel (int): The input channels of the connector.
        out_channel (int): The output channels of the connector.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.bn(self.conv(feature))


@MODELS.register_module()
class ReLUConnector(BaseConnector):
    """General connector which contains a conv layer with BN and ReLU.

    Args:
        in_channel (int): The input channels of the connector.
        out_channel (int): The output channels of the connector.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.relu(self.bn(self.conv(feature)))
