# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class ConvConnector(BaseConnector):
    """Convolution connector which .

    Args:
        in_channel (int): The input channel of the connector.
        out_channel (int): The output channel of the connector.
        kernel_size (int): Size of the convolving kernel. Defaults to 1.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int): Zero-padding added to both sides of the input.
            Defaults to 0.
        use_norm (bool): Whether to use normalization layer.
        use_relu (bool): Whether to use ReLU activation.
        conv_cfg (dict, optional): The config to control the convolution.
        norm_cfg (dict, optional): The config to control the normalization.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        use_norm: bool = False,
        use_relu: bool = False,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.use_norm = use_norm
        self.use_relu = use_relu

        self.conv = build_conv_layer(
            conv_cfg,
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        if use_norm:
            assert norm_cfg, '"use_norm" is True but "norm_cfg is None."'
            _, self.norm = build_norm_layer(norm_cfg, out_channel)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        feature = self.conv(feature)
        if self.use_norm:
            feature = self.norm(feature)
        if self.use_relu:
            feature = self.relu(feature)
        return feature
