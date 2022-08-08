# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.losses.cross_entropy_loss import soft_cross_entropy

from mmrazor.registry import MODELS


@MODELS.register_module()
class KDSoftCELoss(nn.Module):
    """Distilling the Knowledge in a Neural Network, NIPS2014. Based on Soft
    Cross Entropy criterion.

    https://arxiv.org/pdf/1503.02531.pdf

    Args:
        tau (int, optional): Temperature. Defaults to 1.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        mult_tem_square (bool, optional): Multiply square of temperature
            or not. Defaults to True.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        mult_tem_square: bool = True,
        loss_weight: float = 1.0,
    ):
        super(KDSoftCELoss, self).__init__()
        self.tau = tau
        self.mult_tem_square = mult_tem_square
        self.loss_weight = loss_weight
        self.cls_criterion = soft_cross_entropy

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(
        self,
        preds_S,
        preds_T,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        reduction = (
            reduction_override if reduction_override else self.reduction)

        preds_S = preds_S.detach()
        preds_S = preds_S / self.tau
        soft_label = F.softmax((preds_T / self.tau), dim=-1)
        loss_cls = self.loss_weight * self.cls_criterion(
            preds_S,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        if self.mult_tem_square:
            loss_cls *= (self.tau**2)
        return loss_cls
