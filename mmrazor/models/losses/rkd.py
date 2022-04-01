# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def euclidean_distance(pred, squared=False, eps=1e-12):
    """Calculate the Euclidean distance between the two examples in the output
    representation space.

    Args:
        pred (torch.Tensor): The prediction of the teacher or student with
            shape (N, C).
        squared (bool): Whether to calculate the squared Euclidean
            distance. Defaults to False.
        eps (float): The minimum Euclidean distance between the two
            examples. Defaults to 1e-12.
    """
    pred_square = pred.pow(2).sum(dim=-1)  # (N, )
    prod = torch.mm(pred, pred.t())  # (N, N)
    distance = (pred_square.unsqueeze(1) + pred_square.unsqueeze(0) -
                2 * prod).clamp(min=eps)  # (N, N)

    if not squared:
        distance = distance.sqrt()

    distance = distance.clone()
    distance[range(len(prod)), range(len(prod))] = 0
    return distance


def angle(pred):
    """Calculate the angle-wise relational potential which measures the angle
    formed by the three examples in the output representation space.

    Args:
        pred (torch.Tensor): The prediction of the teacher or student with
            shape (N, C).
    """
    pred_vec = pred.unsqueeze(0) - pred.unsqueeze(1)  # (N, N, C)
    norm_pred_vec = F.normalize(pred_vec, p=2, dim=2)
    angle = torch.bmm(norm_pred_vec,
                      norm_pred_vec.transpose(1, 2)).view(-1)  # (N*N*N, )
    return angle


@LOSSES.register_module()
class RelationalKD(nn.Module):
    """PyTorch version of `Relational Knowledge Distillation.

    <https://arxiv.org/abs/1904.05068>`_.
    Args:
        loss_weight_d (float): Weight of distance-wise distillation loss.
            Defaults to 25.0.
        loss_weight_a (float): Weight of angle-wise distillation loss.
            Defaults to 50.0.
        with_l2_norm (bool): Whether to normalize the model predictions before
            calculating the loss. Defaults to True.
    """

    def __init__(self,
                 loss_weight_d=25.0,
                 loss_weight_a=50.0,
                 with_l2_norm=True):
        super(RelationalKD, self).__init__()
        self.loss_weight_d = loss_weight_d
        self.loss_weight_a = loss_weight_a
        self.with_l2_norm = with_l2_norm

    def distance_loss(self, preds_S, preds_T):
        """Calculate distance-wise distillation loss."""
        d_T = euclidean_distance(preds_T, squared=False)
        # mean_d_T is a normalization factor for distance
        mean_d_T = d_T[d_T > 0].mean()
        d_T = d_T / mean_d_T

        d_S = euclidean_distance(preds_S, squared=False)
        mean_d_S = d_S[d_S > 0].mean()
        d_S = d_S / mean_d_S

        return F.smooth_l1_loss(d_S, d_T)

    def angle_loss(self, preds_S, preds_T):
        """Calculate the angle-wise distillation loss."""
        angle_T = angle(preds_T)
        angle_S = angle(preds_S)
        return F.smooth_l1_loss(angle_S, angle_T)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).
        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_S = preds_S.view(preds_S.shape[0], -1)
        preds_T = preds_T.view(preds_T.shape[0], -1)
        if self.with_l2_norm:
            preds_S = F.normalize(preds_S, p=2, dim=-1)
            preds_T = F.normalize(preds_T, p=2, dim=-1)

        loss = 0.
        if self.loss_weight_d > 0:
            loss += self.distance_loss(preds_S, preds_T) * self.loss_weight_d
        if self.loss_weight_a > 0:
            loss += self.angle_loss(preds_S, preds_T) * self.loss_weight_a

        return loss
