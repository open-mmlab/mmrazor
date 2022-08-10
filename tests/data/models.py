# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn import Module
from torch import Tensor
import torch.nn as nn
import torch

# this file includes models for tesing.


class MultiConcatModel(Module):
    """
        x----------------
        |op1    |op2    |op4
        x1      x2      x4
        |       |       |
        |cat-----       |
        cat1            |
        |op3            |
        x3              |
        |cat-------------
        cat2
        |avg_pool
        x_pool
        |fc
        output
    """

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(16, 8, 1)
        self.op4 = nn.Conv2d(3, 8, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1000)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.op3(cat1)
        x4 = self.op4(x)
        cat2 = torch.cat([x3, x4], dim=1)
        x_pool = self.avg_pool(cat2).flatten(1)
        output = self.fc(x_pool)

        return output


class MultiConcatModel2(Module):
    """
        x---------------
        |op1    |op2   |op3
        x1      x2      x3
        |       |       |
        |cat-----       |
        cat1            |
        |cat-------------
        cat2
        |op4
        x4
        |avg_pool
        x_pool
        |fc
        output
    """

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(3, 8, 1)
        self.op4 = nn.Conv2d(24, 8, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 1000)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        x3 = self.op3(x)
        cat1 = torch.cat([x1, x2], dim=1)
        cat2 = torch.cat([cat1, x3], dim=1)
        x4 = self.op4(cat2)

        x_pool = self.avg_pool(x4).reshape([x4.shape[0], -1])
        output = self.fc(x_pool)

        return output


class ConcatModel(Module):
    """
        x------------
        |op1,bn1    |op2,bn2 
        x1          x2 
        |cat--------| 
        cat1 
        |op3 
        x3
        |avg_pool
        x_pool
        |fc
        output
    """

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.op3 = nn.Conv2d(16, 8, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 1000)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bn1(self.op1(x))
        x2 = self.bn2(self.op2(x))
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.op3(cat1)

        x_pool = self.avg_pool(x3).flatten(1)
        output = self.fc(x_pool)

        return output
  

class ResBlock(Module):
    """
        x
        |op1,bn1
        x1-----------
        |op2,bn2    |
        x2          |
        +------------
        |op3
        x3
        |avg_pool
        x_pool
        |fc
        output
    """

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(8, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.op3 = nn.Conv2d(8, 8, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 1000)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bn1(self.op1(x))
        x2 = self.bn2(self.op2(x1))
        x3 = self.op3(x2 + x1)
        x_pool = self.avg_pool(x3).flatten(1)
        output = self.fc(x_pool)
        return output


class LineModel(Module):
    """
        x
        |net0,net1
        |net2
        |net3
        x1
        |fc
        output
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d(1))
        self.linear = nn.Linear(16, 1000)

    def forward(self, x):
        x1 = self.net(x)
        x1 = x1.reshape([x1.shape[0], -1])
        return self.linear(x1)


class AddCatModel(Module):
    """
        x------------------------
        |op1    |op2    |op3    |op4
        x1      x2      x3      x4
        |       |       |       |
        |cat-----       |cat-----
        cat1            cat2
        |               |
        +----------------
        x5
        |avg_pool
        x_pool
        |fc
        y
    """

    def __init__(self) -> None:
        super().__init__()
        self.op1 = nn.Conv2d(3, 2, 3)
        self.op2 = nn.Conv2d(3, 6, 3)
        self.op3 = nn.Conv2d(3, 4, 3)
        self.op4 = nn.Conv2d(3, 4, 3)
        self.op5 = nn.Conv2d(8, 16, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1000)

    def forward(self, x):
        x1 = self.op1(x)
        x2 = self.op2(x)
        x3 = self.op3(x)
        x4 = self.op4(x)

        cat1 = torch.cat((x1, x2), dim=1)
        cat2 = torch.cat((x3, x4), dim=1)
        x5 = self.op5(cat1 + cat2)
        x_pool = self.avg_pool(x5).flatten(1)
        y = self.fc(x_pool)
        return y


class GroupWiseConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.op1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(8, 16, 3, 1, 1, groups=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.op3 = nn.Conv2d(16, 32, 3, 1, 1)

    def forward(self, x):
        x1 = self.op1(x)
        x1 = self.bn1(x1)
        x2 = self.op2(x1)
        x2 = self.bn2(x2)
        x3 = self.op3(x2)
        x_pool = self.avg_pool(x3).flatten(1)
        return self.fc(x_pool)


class Xmodel(nn.Module):
    """
        x--------
        |op1    |op2
        x1      x2
        |       |
        +--------
        x12------
        |op3    |op4
        x3      x4
        |       |
        +--------
        x34
        |avg_pool
        x_pool
        |fc
        y
    """

    def __init__(self) -> None:
        super().__init__()
        self.op1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.op2 = nn.Conv2d(3, 8, 3, 1, 1)
        self.op3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.op4 = nn.Conv2d(8, 16, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1000)

    def forward(self, x):
        x1 = self.op1(x)
        x2 = self.op2(x)
        x12 = x1 * x2
        x3 = self.op3(x12)
        x4 = self.op4(x12)
        x34 = x3 + x4
        x_pool = self.avg_pool(x34).flatten(1)
        return self.fc(x_pool)


class MultipleUseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv_multiple_use = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv_last = nn.Conv2d(16, 32, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32, 1000)

    def forward(self, x):
        xs = [
            conv(x)
            for conv in [self.conv0, self.conv1, self.conv2, self.conv3]
        ]
        xs_ = [self.conv_multiple_use(x_) for x_ in xs]
        x_sum = 0
        for x_ in xs_:
            x_sum = x_sum + x_
        feature = self.conv_last(x_sum)
        pool = self.avg_pool(feature).flatten(1)
        return self.linear(pool)


default_models = [
    LineModel, ResBlock, AddCatModel, ConcatModel, MultiConcatModel,
    MultiConcatModel2, GroupWiseConvModel, Xmodel, MultipleUseModel
]


class ModelLibrary:

    # includes = [
    #     'alexnet',        # pass
    #     'densenet',       # pass
    #     # 'efficientnet',   # pass
    #     # 'googlenet',      # pass.
    #     #   googlenet return a tuple when training,
    #     #   so it should trace in eval mode
    #     # 'inception',      # failed
    #     # 'mnasnet',        # pass
    #     # 'mobilenet',      # pass
    #     # 'regnet',         # failed
    #     # 'resnet',         # pass
    #     # 'resnext',        # failed
    #     # 'shufflenet',     # failed
    #     # 'squeezenet',     # pass
    #     # 'vgg',            # pass
    #     # 'wide_resnet',    # pass
    # ]

    def __init__(self, include=[]) -> None:

        self.include_key = include

        self.model_creator = self.get_torch_models()

    def __repr__(self) -> str:
        s = f'model: {len(self.model_creator)}\n'
        for creator in self.model_creator:
            s += creator.__name__ + '\n'
        return s

    def get_torch_models(self):
        from inspect import isfunction

        import torchvision

        attrs = dir(torchvision.models)
        models = []
        for name in attrs:
            module = getattr(torchvision.models, name)
            if isfunction(module):
                models.append(module)
        return models

    def export_models(self):
        models = []
        for creator in self.model_creator:
            if self.is_include(creator.__name__):
                models.append(creator)
        return models

    def is_include(self, name):
        for key in self.include_key:
            if key in name:
                return True
        return False

    def include(self):
        include = []
        for creator in self.model_creator:
            for key in self.include_key:
                if key in creator.__name__:
                    include.append(creator)
        return include

    def uninclude(self):
        include = self.include()
        uninclude = []
        for creator in self.model_creator:
            if creator not in include:
                uninclude.append(creator)
        return uninclude
