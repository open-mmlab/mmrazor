# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply

from ..builder import LOSSES


@LOSSES.register_module()
class PredictionGuidedFeatureImitation(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(PredictionGuidedFeatureImitation, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, student_neck_outputs, student_cls_scores,
                teacher_neck_outputs, teacher_cls_scores):

        loss_list = multi_apply(self.loss_single, student_neck_outputs,
                                student_cls_scores, teacher_neck_outputs,
                                teacher_cls_scores)[0]

        return sum(loss_list) / len(loss_list) * self.loss_weight

    def loss_single(self, student_neck_output, student_cls_score,
                    teacher_neck_output, teacher_cls_score):
        p_diff = F.mse_loss(
            student_cls_score, teacher_cls_score, reduction='none').mean(dim=1)
        f_diff = F.mse_loss(
            student_neck_output, teacher_neck_output,
            reduction='none').mean(dim=1)
        loss_pfi = (p_diff * f_diff).mean()

        return loss_pfi,
