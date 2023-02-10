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
        use_norm (bool): Whether to normalize the feature with batch
            normalization before calculating the distillation loss. Defaults
            to False.
    """

    def __init__(self, tau=1.0, loss_weight=1.0, use_norm=False):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.use_norm = use_norm
        if use_norm:
            self.norm = None

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

        if self.use_norm:
            if self.norm is None:
                self.norm = nn.BatchNorm2d(
                    C, affine=False, track_running_stats=False)
            preds_S, preds_T = self.norm(preds_S), self.norm(preds_T)

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss
