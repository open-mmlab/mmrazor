# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch
from mmengine.model import BaseModel
from torch import nn

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
class ToyOFDStudent(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(100)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            out = self.bn(self.conv(batch_inputs))
            return dict(loss=out)
        elif mode == 'predict':
            out = self.bn(self.conv(batch_inputs) + 1)
            return out
        elif mode == 'tensor':
            out = self.bn(self.conv(batch_inputs) + 2)
            return out


@MODELS.register_module()
class ToyOFDTeacher(ToyOFDStudent):

    def __init__(self):
        super().__init__()


@dataclass(frozen=True)
class Data:
    latent_dim: int = 1


@MODELS.register_module()
class ToyGenerator(BaseModel):

    def __init__(self, latent_dim=4, out_channel=3):
        super().__init__(data_preprocessor=None, init_cfg=None)
        self.latent_dim = latent_dim
        self.out_channel = out_channel
        self.conv = nn.Conv2d(self.latent_dim, self.out_channel, 1)

        # Imitate the structure of generator in separate model_wrapper.
        self.module = Data(latent_dim=self.latent_dim)

    def forward(self, data=None, batch_size=4):
        fakeimg_init = torch.randn(batch_size, self.latent_dim, 1, 1)
        fakeimg = self.conv(fakeimg_init)
        return fakeimg


@MODELS.register_module()
class ToyDistillLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, arg1, arg2):
        return arg1 + arg2
