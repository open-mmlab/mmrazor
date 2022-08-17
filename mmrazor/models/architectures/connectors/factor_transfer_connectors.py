# Copyright (c) OpenMMLab. All rights reserved.
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
    """

    def __init__(self, in_channel, out_channel, use_bn=False, **kwargs):

        super(Paraphraser, self).__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward_train(self, x):
        with torch.no_grad():
            factor = self.encoder(x)
        return factor

    def forward_pretrain(self, s_feat, t_feat):
        factor = self.encoder(t_feat)
        t_feat_rec = self.decoder(factor)

        return self.pretrain_criterion(t_feat, t_feat_rec)


@MODELS.register_module()
class Translator(BaseConnector):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer,
    NeurIPS 2018. https://arxiv.org/pdf/1802.04977.pdf.

    student connector of FT.

    Args:
        in_channel ([int]): number of input channels.
        out_channel ([int]): number of output channels.
        use_bn (bool, optional): Defaults to False.
    """

    def __init__(self, in_channel, out_channel, use_bn=True, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward_train(self, x):
        return self.encoder(x)
