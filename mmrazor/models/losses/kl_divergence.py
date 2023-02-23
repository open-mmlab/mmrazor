# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.losses.utils import weighted_loss
from mmrazor.registry import MODELS


@weighted_loss
def kl_div(preds_S, preds_T, tau: float = 1.0):
    """Calculate the KL divergence between `preds_S` and `preds_T`.

    Args:
        preds_S (torch.Tensor): The student model prediction with shape (N, C).
        preds_T (torch.Tensor): The teacher model prediction with shape (N, C).
        tau (float): Temperature coefficient.
    """
    softmax_pred_T = F.softmax(preds_T / tau, dim=1)
    logsoftmax_preds_S = F.log_softmax(preds_S / tau, dim=1)
    loss = (tau**2) * F.kl_div(
        logsoftmax_preds_S, softmax_pred_T, reduction='none')
    return loss


@MODELS.register_module()
class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self,
                preds_S,
                preds_T,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean", "sum" and "batchmean".

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum', 'batchmean')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.teacher_detach:
            preds_T = preds_T.detach()
        loss = kl_div(
            preds_S,
            preds_T,
            tau=self.tau,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * loss
