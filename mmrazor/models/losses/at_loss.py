# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class ATLoss(nn.Module):
    """"Paying More Attention to Attention: Improving the Performance of
    Convolutional Neural Networks via Attention Transfer" Conference paper at
    ICLR2017 https://openreview.net/forum?id=Sks9_ajex.

    https://github.com/szagoruyko/attention-transfer/blob/master/utils.py

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """"Forward function for ATLoss."""
        loss = (self.calc_attention_matrix(s_feature) -
                self.calc_attention_matrix(t_feature)).pow(2).mean()
        return self.loss_weight * loss

    def calc_attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """"Calculate the attention matrix.

        Args:
            x (torch.Tensor): Input features.
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
