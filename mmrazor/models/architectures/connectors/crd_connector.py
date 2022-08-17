# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class CRDConnector(BaseConnector):
    """Connector with linear layer.

    Args:
        dim_in (int, optional): input channels. Defaults to 1024.
        dim_out (int, optional): output channels. Defaults to 128.
    """

    def __init__(self,
                 dim_in: int = 1024,
                 dim_out: int = 128,
                 **kwargs) -> None:
        super(CRDConnector, self).__init__(**kwargs)
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer.

    Args:
        power (int, optional): power. Defaults to 2.
    """

    def __init__(self, power: int = 2) -> None:
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
