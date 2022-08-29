# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class OFDLoss(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation
    https://sites.google.com/view/byeongho-heo/overhaul.

    The partial L2loss, only calculating loss when
    `out_s > out_t` or `out_t > 0`.

    Args:
        loss_weight (float, optional): loss weight. Defaults to 1.0.
        mul_factor (float, optional): multiply factor. Defaults to 1000.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 mul_factor: float = 1000.) -> None:
        super(OFDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.mul_factor = mul_factor

    def forward_train(self, s_feature: torch.Tensor,
                      t_feature: torch.Tensor) -> torch.Tensor:
        """forward func for training.

        Args:
            s_feature (torch.Tensor): student's feature
            t_feature (torch.Tensor): teacher's feature

        Returns:
            torch.Tensor: loss
        """
        bsz = s_feature.shape[0]
        loss = torch.nn.functional.mse_loss(
            s_feature, t_feature, reduction='none')
        loss = loss * ((s_feature > t_feature) | (t_feature > 0)).float()
        return loss.sum() / bsz / self.mul_factor

    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        """forward func.

        Args:
            s_feature (torch.Tensor): student's feature
            t_feature (torch.Tensor): teacher's feature

        Returns:
            torch.Tensor: loss
        """
        return self.loss_weight * self.forward_train(s_feature, t_feature)
