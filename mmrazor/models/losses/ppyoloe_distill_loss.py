# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.models import IoULoss

from mmrazor.registry import MODELS


def quality_focal_loss(pred_logits,
                       soft_target_logits,
                       beta=2.0,
                       num_total_pos=None):
    soft_target = soft_target_logits.sigmoid()
    pred_sigmoid = pred_logits.sigmoid()
    scale_factor = pred_sigmoid - soft_target
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, soft_target,
        reduction='none') * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)

    if num_total_pos is not None:
        loss = loss.sum() / num_total_pos
    else:
        loss = loss.mean()
    return loss


def distribution_focal_loss(pred_corners, target_corners, weight_targets=None):
    target_corners_label = F.softmax(target_corners, dim=-1)
    loss_dfl = F.cross_entropy(
        pred_corners, target_corners_label, reduction='none')
    # loss_dfl = loss_dfl.sum(dim=1, keepdim=False)
    if weight_targets is not None:
        loss_dfl = loss_dfl * weight_targets
        # loss_dfl = loss_dfl * (weight_targets.expand([-1, 4]).reshape([-1]))
        loss_dfl = loss_dfl.sum(-1) / weight_targets.sum()
    else:
        loss_dfl = loss_dfl.mean(-1)
    return loss_dfl / 4.0  # 4 direction


@MODELS.register_module()
class QualityFocalLoss(nn.Module):

    def __init__(self, beta=2.0, loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T, num_total_pos=None):
        return self.loss_weight * quality_focal_loss(
            preds_S, preds_T, beta=self.beta, num_total_pos=num_total_pos)


@MODELS.register_module()
class DistributionFocalLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T, weight_targets=None):
        loss_cls = self.loss_weight * distribution_focal_loss(
            preds_S, preds_T, weight_targets)
        return loss_cls


@MODELS.register_module()
class BboxLoss(nn.Module):

    def __init__(self, loss_weight=10.0):
        super(BboxLoss, self).__init__()
        self.loss_bbox = IoULoss(
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='none',
            loss_weight=loss_weight,
            return_iou=False)

    def forward(self, s_bbox, t_bbox, weight_targets=None):
        if weight_targets is not None:
            loss = torch.sum(self.loss_bbox(s_bbox, t_bbox) * weight_targets)
            avg_factor = weight_targets.sum()
            loss = loss / avg_factor
        else:
            loss = torch.mean(self.loss_bbox(s_bbox, t_bbox))
        return loss


@MODELS.register_module()
class MainKDLoss(nn.Module):

    def __init__(self, loss_weight=1.0, tau=1.0):
        super(MainKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.tau = tau

    def kl_div(self, preds_S, preds_T):
        preds_T = preds_T.detach()

        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction='none')
        return loss

    def forward(self, mask_positive, pred_scores, soft_cls):
        num_classes = soft_cls.size(-1)
        num_pos = mask_positive.sum()
        if num_pos > 0:
            cls_mask = mask_positive.unsqueeze(-1).repeat([1, 1, num_classes])
            pred_scores_pos = torch.masked_select(
                pred_scores, cls_mask).reshape([-1, num_classes])
            soft_cls_pos = torch.masked_select(soft_cls, cls_mask).reshape(
                [-1, num_classes])
            loss_kd = self.kl_div(pred_scores_pos, soft_cls_pos)
            # loss_kd = loss_kd.sum(dim=1)

            avg_factor = num_pos
            loss_kd = loss_kd.sum() / avg_factor
        else:
            loss_kd = torch.zeros([1])
        return loss_kd
