# Copyright (c) OpenMMLab. All rights reserved.
from math import log
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from ..ops.darts_series import DartsSepConv
from .base_connector import BaseConnector


@MODELS.register_module()
class BYOTConnector(BaseConnector):
    """BYOTConnector connector that adds a self-attention with DartsSepConv.

    Args:
        in_channel (int): The input channel of the DartsSepConv.
            Use like input_tensor_channel = in_channel * expansion.
        out_channel (int): The output channel of the DartsSepConv.
            Use like output_tensor_channel = out_channel * expansion.
        num_classes (int): The classification class num.
        expansion (int): Expansion of DartsSepConv. Default to 4.
        pool_size (int | tuple[int]): Average 2D pool size. Default to 4.
        kernel_size (int | tuple[int]): Size of the convolving kernel in
            DartsSepConv. Same as that in ``nn._ConvNd``. Default to 3.
        stride (int | tuple[int]): Stride of the first layer in DartsSepConv.
            Same as that in ``nn._ConvNd``. Default to 1.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        num_classes: int,
        expansion: int = 4,
        pool_size: Union[int, Tuple[int]] = 4,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.attention = nn.Sequential(
            DartsSepConv(
                in_channels=in_channel * expansion,
                out_channels=in_channel * expansion,
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
                    in_channels=_in_channel * expansion,
                    out_channels=_in_channel * 2 * expansion,
                    kernel_size=kernel_size,
                    stride=stride))
            _in_channel *= 2
        scala.append(nn.AvgPool2d(pool_size))
        self.scala = nn.Sequential(*scala)
        self.fc = nn.Linear(out_channel * expansion, num_classes)

    def forward_train(self, feature: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        feat = self.attention(feature)
        feat = feat * feature

        feat = self.scala(feat)
        feat = feat.view(feature.size(0), -1)
        logits = self.fc(feat)
        return (feat, logits)
