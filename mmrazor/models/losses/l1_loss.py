# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class L1Loss(nn.Module):
    """Calculate the one-norm loss between the two inputs.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By
            default, the losses are averaged over each loss element in the
            batch. Note that for some losses, there multiple elements per
            sample. If the field :attr:`size_average` is set to ``False``, the
            losses are instead summed for each minibatch. Ignored when reduce
            is ``False``. Defaults to True.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By
            default, the losses are averaged or summed over observations for
            each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element
            instead and ignores :attr:`size_average`. Defaults to True.
        reduction (string, optional): Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no
            reduction will be applied, ``'mean'``: the sum of the output will
            be divided by the number of elements in the output, ``'sum'``: the
            output will be summed. Note: :attr:`size_average` and
            :attr: `reduce` are in the process of being deprecated, and in the
            meantime, specifying either of those two args will override
            :attr:`reduction`. Defaults to mean.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.size_average = size_average
        self.reduce = reduce

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        loss = F.l1_loss(s_feature, t_feature, self.size_average, self.reduce,
                         self.reduction)
        return self.loss_weight * loss
