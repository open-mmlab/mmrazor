# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from mmengine.model import BaseModule

from mmrazor.models.utils import get_module_device


class BaseGenerator(BaseModule):
    """The base class for generating images.

    Args:
        img_size (int): The size of generated image.
        latent_dim (int): The dimension of latent data.
        hidden_channels (int): The dimension of hidden channels.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 img_size: int,
                 latent_dim: int,
                 hidden_channels: int,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

    def process_latent(self,
                       latent_data: Optional[torch.Tensor] = None,
                       batch_size: int = 1) -> torch.Tensor:
        """Generate the latent data if the input is None. Put the latent data
        into the current gpu.

        Args:
            latent_data (torch.Tensor, optional): The latent data. Defaults to
                None.
            batch_size (int): The batch size of the latent data. Defaults to 1.
        """
        if isinstance(latent_data, torch.Tensor):
            assert latent_data.shape[1] == self.latent_dim, \
                'Second dimension of the input must be equal to "latent_dim",'\
                f'but got {latent_data.shape[1]} != {self.latent_dim}.'
            if latent_data.ndim == 2:
                batch_data = latent_data
            else:
                raise ValueError('The noise should be in shape of (n, c)'
                                 f'but got {latent_data.shape}')
        elif latent_data is None:
            assert batch_size > 0, \
                '"batch_size" should larger than zero when "latent_data" is '\
                f'None, but got {batch_size}.'
            batch_data = torch.randn((batch_size, self.latent_dim))

        # putting data on the right device
        batch_data = batch_data.to(get_module_device(self))
        return batch_data

    def forward(self) -> None:
        """Forward function."""
        raise NotImplementedError
