# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class CRDConnector(BaseConnector):
    """Connector with linear layer.

    Args:
        dim_in ([int])
        dim_out ([int])
    """

    def __init__(self, dim_in=1024, dim_out=128, **kwargs):
        super(CRDConnector, self).__init__(**kwargs)
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward_train(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer."""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
