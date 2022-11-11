# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        preds_T = preds_T.detach()
        loss = F.cross_entropy(preds_S, preds_T.argmax(dim=1))
        return loss * self.loss_weight
