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
    """
    def __init__(
        self,
        tau: float = 1.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
    ):
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

    def forward(self, preds_S, preds_T, data_samples):
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            labels_GT = torch.stack([i.gt_label.score for i in data_samples])
        else:
            labels_GT = torch.hstack([i.gt_label.label for i in data_samples])
        gt_mask = self._get_gt_mask(preds_S, labels_GT)
        tckd_loss = self.get_tckd_loss(preds_S, preds_T, labels_GT, gt_mask)
        nckd_loss = self.get_nckd_loss(preds_S, preds_T, gt_mask)
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return self.loss_weight * loss

    def get_nckd_loss(self, preds_S, preds_T, mask_GT):
        # non-target class knowledge distillation
        s_nckd = F.log_softmax(preds_S / self.tau - 1000.0 * mask_GT, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - 1000.0 * mask_GT, dim=1)
        return self.kl_loss(s_nckd, t_nckd)

    def get_tckd_loss(self, preds_S, preds_T, labels_GT, mask_GT):
        # target class knowledge distillation
        other_mask = self._get_non_gt_mask(preds_S, labels_GT)
        s_tckd = F.softmax(preds_S / self.tau, dim=1)
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_tckd, mask_GT, other_mask))
        mask_teacher = self._cat_mask(t_tckd, mask_GT, other_mask)
        return self.kl_loss(mask_student, mask_teacher)

    def kl_loss(self, preds_S, preds_T):
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau**2
        if self.reduction != 'batchmean':
            kl_loss /= preds_S.shape[0]
        return kl_loss

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(self, logits, target):

        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()
