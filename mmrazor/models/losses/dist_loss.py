# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


@MODELS.register_module()
class DISTLoss(nn.Module):

    def __init__(
        self,
        beta=1.0,
        gamma=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, preds_S, preds_T: torch.Tensor):
        if self.teacher_detach:
            preds_T = preds_T.detach()
        y_s = (preds_S / self.tau).softmax(dim=1)
        y_t = (preds_T / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss * self.loss_weight
