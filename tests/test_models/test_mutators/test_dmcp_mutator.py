# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import *  # noqa: F401,F403
from torch import Tensor, nn
from torch.nn import Module

from mmrazor.models.mutators import DMCPChannelMutator


class ResBlock(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(8, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.op3 = nn.Conv2d(8, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bn1(self.op1(x))
        x2 = self.bn2(self.op2(x1))
        x3 = self.op3(x2 + x1)
        return x3


def test_DMCP_channel_mutator() -> None:
    imgs = torch.randn(16, 3, 224, 224)

    # ResBlock
    mutator = DMCPChannelMutator(channel_unit_cfg=dict(type='DMCPChannelUnit'))

    model = ResBlock()
    mutator.prepare_from_supernet(model)
    for mode in ['max', 'min', 'random', 'expected', 'direct']:
        mutator.sample_subnet(mode, arch_train=True)
    out3 = model(imgs)

    assert out3.shape == (16, 8, 224, 224)
