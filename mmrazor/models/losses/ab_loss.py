# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class ABLoss(nn.Module):
    """Activation Boundaries Loss.

    Paper: Knowledge Transfer via Distillation of Activation Boundaries
    Formed by Hidden Neurons, AAAI2019. https://arxiv.org/pdf/1811.03233.pdf

    Modified from: https://github.com/facebookresearch/AlphaNet

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        margin (float): Relaxation for training stability. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.margin = margin

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """ABLoss forward function.

        Args:
            s_features (torch.Tensor): Student featuremap.
            t_features (torch.Tensor): Teacher featuremap.
        """
        batch_size = s_feature.shape[0]
        loss = self.criterion_alternative_l2(s_feature, t_feature)
        loss = loss / batch_size / 1000 * 3
        return self.loss_weight * loss

    def criterion_alternative_l2(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Piecewise differentiable loss approximating the activation
        boundaries loss.

        Guide the student learns a separating boundary between activation
        region and deactivation region formed by each neuron in the teacher.

        Args:
            source (torch.Tensor): Student featuremap.
            target (torch.Tensor): Teacher featuremap.
        """
        loss = ((source + self.margin)**2 * ((source > -self.margin) &
                                             (target <= 0)).float() +
                (source - self.margin)**2 * ((source <= self.margin) &
                                             (target > 0)).float())
        return torch.abs(loss).sum()
