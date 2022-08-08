# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation, CVPR2022.

    link: https://arxiv.org/abs/2203.08679
    reformulate the classical KD loss into two parts:
        1. target class knowledge distillation (TCKD)
        2. non-target class knowledge distillation (NCKD).
    Args:
    tau (float): Temperature coefficient. Defaults to 4.0.
    alpha (float): Weight of TCKD loss. Defaults to 1.0.
    beta (float): Weight of NCKD loss. Defaults to 8.0.
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
    """

    def __init__(
        self,
        tau: float = 4.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
    ) -> None:
        super(DKDLoss, self).__init__()
        self.tau = tau
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        data_samples: list,
    ) -> torch.Tensor:
        """DKDLoss forward function.

        Args:
            student (torch.Tensor): Student featuremap.
            teacher (torch.Tensor): Teacher featuremap.
            data_samples (list): List of model input sample data,
            contains gt_label (dict of label and one-hot score)
        """
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            gt_labels = torch.stack([i.gt_label.score for i in data_samples])
        else:
            gt_labels = torch.hstack([i.gt_label.label for i in data_samples])
        gt_mask = self._get_gt_mask(student, gt_labels)
        tckd_loss = self._get_tckd_loss(student, teacher, gt_labels, gt_mask)
        nckd_loss = self._get_nckd_loss(student, teacher, gt_mask)
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return self.loss_weight * loss

    def _get_nckd_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clac non-target class knowledge distillation."""
        s_nckd = F.log_softmax(student / self.tau - 1000.0 * gt_mask, dim=1)
        t_nckd = F.softmax(teacher / self.tau - 1000.0 * gt_mask, dim=1)
        return self.kl_loss(s_nckd, t_nckd)

    def _get_tckd_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calc target class knowledge distillation."""
        other_mask = self._get_non_gt_mask(student, gt_labels)
        s_tckd = F.softmax(student / self.tau, dim=1)
        t_tckd = F.softmax(teacher / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_tckd, gt_mask, other_mask))
        mask_teacher = self._cat_mask(t_tckd, gt_mask, other_mask)
        return self._kl_loss(mask_student, mask_teacher)

    def _kl_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        kl_loss = F.kl_div(
            student, teacher, size_average=False,
            reduction=self.reduction) * self.tau**2
        return kl_loss

    def _cat_mask(
        self,
        target: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> torch.Tensor:
        t1 = (target * mask1).sum(dim=1, keepdims=True)
        t2 = (target * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()
