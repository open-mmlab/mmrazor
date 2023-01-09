# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation.

    <https://arxiv.org/abs/2205.01529>`

    Args:
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
    """

    def __init__(self, alpha_mgd: float = 0.00002) -> None:
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.loss_mse = nn.MSELoss(reduction='sum')

    def forward(self, preds_S: torch.Tensor,
                preds_T: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape == preds_T.shape
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S: torch.Tensor,
                     preds_T: torch.Tensor) -> torch.Tensor:
        """Get MSE distance of preds_S and preds_T.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated mse distance value.
        """
        N, C, H, W = preds_T.shape
        dis_loss = self.loss_mse(preds_S, preds_T) / N

        return dis_loss
