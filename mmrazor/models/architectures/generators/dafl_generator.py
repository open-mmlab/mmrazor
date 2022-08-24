# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from .base_generator import BaseGenerator


@MODELS.register_module()
class DAFLGenerator(BaseGenerator):
    """Generator for DAFL.

    Args:
        img_size (int): The size of generated image.
        latent_dim (int): The dimension of latent data.
        hidden_channels (int): The dimension of hidden channels.
        scale_factor (int, optional): The scale factor for F.interpolate.
            Defaults to 2.
        bn_eps (float, optional): The eps param in bn. Defaults to 0.8.
        leaky_slope (float, optional): The slope param in leaky relu. Defaults
            to 0.2.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        img_size: int,
        latent_dim: int,
        hidden_channels: int,
        scale_factor: int = 2,
        bn_eps: float = 0.8,
        leaky_slope: float = 0.2,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            img_size, latent_dim, hidden_channels, init_cfg=init_cfg)
        self.init_size = self.img_size // (scale_factor**2)
        self.scale_factor = scale_factor
        self.linear = nn.Linear(self.latent_dim,
                                self.hidden_channels * self.init_size**2)

        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(self.hidden_channels, eps=bn_eps),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels // 2,
                3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(self.hidden_channels // 2, eps=bn_eps),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(self.hidden_channels // 2, 3, 3, stride=1, padding=1),
            nn.Tanh(), nn.BatchNorm2d(3, affine=False))

    def forward(self,
                data: Optional[torch.Tensor] = None,
                batch_size: int = 1) -> torch.Tensor:
        """Forward function for generator.

        Args:
            data (torch.Tensor, optional): The input data. Defaults to None.
            batch_size (int): Batch size. Defaults to 1.
        """
        batch_data = self.process_latent(data, batch_size)
        img = self.linear(batch_data)
        img = img.view(img.shape[0], self.hidden_channels, self.init_size,
                       self.init_size)
        img = self.bn1(img)
        img = F.interpolate(img, scale_factor=self.scale_factor)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=self.scale_factor)
        img = self.conv_blocks2(img)
        return img
