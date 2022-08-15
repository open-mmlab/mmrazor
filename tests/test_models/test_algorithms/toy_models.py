# Copyright (c) OpenMMLab. All rights reserved.

from torch import nn

from mmengine.model import BaseModel
from mmrazor.registry import MODELS


@MODELS.register_module()
class ToyStudent(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            out = self.conv(batch_inputs)
            return dict(loss=out)
        elif mode == 'predict':
            out = self.conv(batch_inputs) + 1
            return out
        elif mode == 'tensor':
            out = self.conv(batch_inputs) + 2
            return out


@MODELS.register_module()
class ToyTeacher(ToyStudent):

    def __init__(self):
        super().__init__()


@MODELS.register_module()
class ToyDistillLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, arg1, arg2):
        return arg1 + arg2
