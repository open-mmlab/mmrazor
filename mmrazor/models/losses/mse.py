# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class MSE(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight=1.0,
        input_channel=1.0,
    ):
        super(MSE, self).__init__()
        self.loss_weight = loss_weight
        self.bn_T = nn.BatchNorm2d(input_channel, affine=False)
        self.bn_S = nn.BatchNorm2d(input_channel, affine=False)
        self.mse = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        preds_T = self.bn_T(preds_T)
        preds_S = self.bn_S(preds_S)

        loss = self.mse(preds_T, preds_S)

        loss = self.loss_weight * loss

        return loss
