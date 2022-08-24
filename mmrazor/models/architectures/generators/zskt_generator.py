# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_generator import BaseGenerator


class View(nn.Module):
    """Class for view tensors.

    Args:
        size (Tuple[int, ...]): Size of the output tensor.
    """

    def __init__(self, size: Tuple[int, ...]) -> None:
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """"Forward function for view tensors."""
        return tensor.view(self.size)


@MODELS.register_module()
class ZSKTGenerator(BaseGenerator):
    """Generator for ZSKT. code link:
    https://github.com/polo5/ZeroShotKnowledgeTransfer/

    Args:
        img_size (int): The size of generated image.
        latent_dim (int): The dimension of latent data.
        hidden_channels (int): The dimension of hidden channels.
        scale_factor (int, optional): The scale factor for F.interpolate.
            Defaults to 2.
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
        leaky_slope: float = 0.2,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            img_size, latent_dim, hidden_channels, init_cfg=init_cfg)
        self.init_size = self.img_size // (scale_factor**2)
        self.scale_factor = scale_factor

        self.layers = nn.Sequential(
            nn.Linear(self.latent_dim,
                      self.hidden_channels * self.init_size**2),
            View((-1, self.hidden_channels, self.init_size, self.init_size)),
            nn.BatchNorm2d(self.hidden_channels),
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                stride=1,
                padding=1), nn.BatchNorm2d(self.hidden_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels // 2,
                3,
                stride=1,
                padding=1), nn.BatchNorm2d(self.hidden_channels // 2),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(self.hidden_channels // 2, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3, affine=True))

    def forward(self,
                data: Optional[torch.Tensor] = None,
                batch_size: int = 1) -> torch.Tensor:
        """Forward function for generator.

        Args:
            data (torch.Tensor, optional): The input data. Defaults to None.
            batch_size (int): Batch size. Defaults to 1.
        """
        batch_data = self.process_latent(data, batch_size)
        return self.layers(batch_data)
