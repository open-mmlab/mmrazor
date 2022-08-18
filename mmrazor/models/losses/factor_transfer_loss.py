# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class FTLoss(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer,
    NeurIPS 2018.

    https://arxiv.org/pdf/1802.04977.pdf
    """

    def __init__(self, loss_weight=1.0):
        super(FTLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward_train(self, s_feature, t_feature):
        loss = self.criterion(self.factor(s_feature), self.factor(t_feature))
        return loss

    def forward(self, s_feature, t_feature):
        return self.loss_weight * self.forward_train(s_feature, t_feature)

    def factor(self, x):
        return F.normalize(x.view(x.size(0), -1))
