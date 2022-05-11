# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

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
        N, C, H, W = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss
