# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class Paraphraser(BaseConnector):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer,
    NeurIPS 2018. https://arxiv.org/pdf/1802.04977.pdf.

    teacher connector of FT.

    Args:
        in_channel ([int]): number of input channels.
        out_channel ([int]): number of output channels.
        use_bn (bool, optional): Defaults to False.
        phase (str, optional): Training phase. Defaults to 'pretrain'.
        use_bn (Optional[bool], optional): use BN or not. Defaults to False.
        init_cfg (Optional[Dict], optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 phase='pretrain',
                 use_bn: Optional[bool] = False,
                 init_cfg: Optional[Dict] = None) -> None:

        super(Paraphraser, self).__init__(init_cfg)
        self._build_modules(in_channel, out_channel, use_bn)

        assert phase in ['pretrain', 'train'], f'Unexpect `phase`: {phase}'
        self.phase = phase

    def _build_modules(self,
                       in_channel: int,
                       out_channel: int,
                       use_bn: Optional[bool] = False) -> None:
        """A helper func to build internal modules.

        Args:
            in_channel (int): input channels
            out_channel (int): output channels
            use_bn (Optional[bool], optional): use BN or not.
                Defaults to False.
        """

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True))

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Forward func for training."""
        with torch.no_grad():
            factor = self.encoder(x)
        return factor

    def forward_pretrain(self, t_feat: torch.Tensor) -> torch.Tensor:
        """Forward func for pretraining."""
        factor = self.encoder(t_feat)
        t_feat_rec = self.decoder(factor)

        return t_feat_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Omitted."""
        if self.phase == 'train':
            return self.forward_train(x)
        elif self.phase == 'pretrain':
            return self.forward_pretrain(x)
        else:
            raise NotImplementedError(
                f'phase: `{self.phase}` is not supported.')


@MODELS.register_module()
class Translator(BaseConnector):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer,
    NeurIPS 2018. https://arxiv.org/pdf/1802.04977.pdf.

    student connector of FT.

    Args:
        in_channel ([int]): number of input channels.
        out_channel ([int]): number of output channels.
        use_bn (bool, optional): Defaults to False.
        init_cfg (Optional[Dict], optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 use_bn: Optional[bool] = True,
                 init_cfg: Optional[Dict] = None) -> None:
        super(Translator, self).__init__(init_cfg)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True))

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Forward func for training."""
        return self.encoder(x)
