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
        x = self.op1(x)
        x = self.bn1(x)
        x = self.op2(x)
        x = self.bn2(x)
        x = self.op3(x)
        return x


default_models = [LineModel, ResBlock, AddCatModel,
                  ConcatModel, MultiConcatModel, MultiConcatModel2, GroupWiseConvModel]
