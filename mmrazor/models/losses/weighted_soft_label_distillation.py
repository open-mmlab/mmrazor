# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class WSLD(nn.Module):
    """PyTorch version of `Rethinking Soft Labels for Knowledge
    Distillation: A Bias-Variance Tradeoff Perspective
    <https://arxiv.org/abs/2102.00650>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        num_classes (int): Defaults to 1000.
    """

    def __init__(self, tau=1.0, loss_weight=1.0, num_classes=1000):
        super(WSLD, self).__init__()

        self.tau = tau
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, student, teacher, gt_labels):

        student_logits = student / self.tau
        teacher_logits = teacher / self.tau

        teacher_probs = self.softmax(teacher_logits)

        ce_loss = -torch.sum(
            teacher_probs * self.logsoftmax(student_logits), 1, keepdim=True)

        student_detach = student.detach()
        teacher_detach = teacher.detach()
        log_softmax_s = self.logsoftmax(student_detach)
        log_softmax_t = self.logsoftmax(teacher_detach)
        one_hot_labels = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        ce_loss_s = -torch.sum(one_hot_labels * log_softmax_s, 1, keepdim=True)
        ce_loss_t = -torch.sum(one_hot_labels * log_softmax_t, 1, keepdim=True)

        focal_weight = ce_loss_s / (ce_loss_t + 1e-7)
        ratio_lower = torch.zeros_like(focal_weight)
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(-focal_weight)
        ce_loss = focal_weight * ce_loss

        loss = (self.tau**2) * torch.mean(ce_loss)

        loss = self.loss_weight * loss

        return loss
