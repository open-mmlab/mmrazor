# Copyright (c) OpenMMLab. All rights reserved.
# this file includes models for tesing.
import math
from torch.nn import Module
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmrazor.models.architectures.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d, DynamicLinear, DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import MutableChannelContainer
from mmrazor.models.mutables import MutableChannelUnit
from mmrazor.models.mutables import DerivedMutable
from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutables import OneShotMutableChannelUnit, OneShotMutableChannel
# this file includes models for tesing.


class arrange_subnet(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        add = torch.arange(x.shape[-1]).unsqueeze(0)
        # add = torch.arange(1000).unsqueeze(0)
        end = torch.add(x, add)
        return end


class UnTracableModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d(1))
        self.linear = nn.Linear(16, 1000)
        self.end = arrange_subnet()

    def forward(self, x):
        x1 = self.net(x)
        x1 = x1.reshape([x1.shape[0], -1])
        logit = self.linear(x1)
        return self.end(logit)


class ConvAttnModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.head = LinearHead(16, 1000)

    def forward(self, x):
        x1 = self.conv(x)
        attn = F.sigmoid(self.pool(x1))
        x_attn = x1 * attn
        x_last = self.conv2(x_attn)
        return self.head(x_last)


class LinearHead(Module):

    def __init__(self, in_channel, num_class=1000) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channel, num_class)

    def forward(self, x):
        pool = self.pool(x).flatten(1)
        return self.linear(pool)


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


class SingleLineModel(nn.Module):
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
    """
        x
        |op1,bn1
        x1
        |op2,bn2
        x2
        |op3
        x3
        |avg_pool
        x_pool
        |fc
        y
    """

    def __init__(self) -> None:
        super().__init__()
        self.op1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(8, 16, 3, 1, 1, groups=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.op3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1000)

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
    """
        x------------------------
        |conv0  |conv1  |conv2  |conv3
        xs.0    xs.1    xs.2    xs.3
        |convm  |convm  |convm  |convm
        xs_.0   xs_.1   xs_.2   xs_.3
        |       |       |       |
        +------------------------
        |
        x_sum
        |conv_last
        feature
        |avg_pool
        pool
        |linear
        output
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv_multiple_use = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv_last = nn.Conv2d(16 * 4, 32, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32, 1000)

    def forward(self, x):
        xs = [
            conv(x)
            for conv in [self.conv0, self.conv1, self.conv2, self.conv3]
        ]
        xs_ = [self.conv_multiple_use(x_) for x_ in xs]
        x_cat = torch.cat(xs_, dim=1)
        feature = self.conv_last(x_cat)
        pool = self.avg_pool(feature).flatten(1)
        return self.linear(pool)


class IcepBlock(nn.Module):
    """
        x------------------------
        |op1    |op2    |op3    |op4
        x1      x2      x3      x4
        |       |       |       |
        cat----------------------
        |
        y_
    """

    def __init__(self, in_c=3, out_c=32) -> None:
        super().__init__()
        self.op1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.op２ = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.op３ = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.op4 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        # self.op5 = nn.Conv2d(out_c*4, out_c, 3)

    def forward(self, x):
        x1 = self.op1(x)
        x2 = self.op2(x)
        x3 = self.op3(x)
        x4 = self.op4(x)
        y_ = [x1, x2, x3, x4]
        y_ = torch.cat(y_, 1)
        return y_


class Icep(nn.Module):

    def __init__(self, num_icep_blocks=2) -> None:
        super().__init__()
        self.icps = nn.Sequential(*[
            IcepBlock(32 * 4 if i != 0 else 3, 32)
            for i in range(num_icep_blocks)
        ])
        self.op = nn.Conv2d(32 * 4, 32, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1000)

    def forward(self, x):
        y_ = self.icps(x)
        y = self.op(y_)
        pool = self.avg_pool(y).flatten(1)
        return self.fc(pool)


class ExpandLineModel(Module):
    """
        x
        |net0,net1,net2
        |net3,net4
        x1
        |fc
        output
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d(2))
        self.linear = nn.Linear(64, 1000)

    def forward(self, x):
        x1 = self.net(x)
        x1 = x1.reshape([x1.shape[0], -1])
        return self.linear(x1)


class MultiBindModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 8, 3, 1, 1)
        self.head = LinearHead(8, 1000)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x12 = x1 + x2
        x3 = self.conv3(x12)
        x123 = x12 + x3
        return self.head(x123)


class DwConvModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 48, 3, 1, 1, groups=48), nn.BatchNorm2d(48),
            nn.ReLU())
        self.head = LinearHead(48, 1000)

    def forward(self, x):
        return self.head(self.net(x))


class SelfAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 4, 4, 4)

        self.num_head = 4
        self.qkv = nn.Linear(32, 32 * 3)
        self.proj = nn.Linear(32, 32)

        self.head = LinearHead(32, 1000)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        h, w = x.shape[-2:]
        x = self._to_token(x)
        x = x + self._forward_attention(x)
        x = self._to_img(x, h, w)
        return self.head(x)

    def _to_img(self, x, h, w):
        x = x.reshape([x.shape[0], h, w, x.shape[2]])
        x = x.permute(0, 3, 1, 2)
        return x

    def _to_token(self, x):
        x = x.flatten(2).transpose(-1, -2)
        return x

    def _forward_attention(self, x: torch.Tensor):
        qkv = self.qkv(x)
        qkv = qkv.reshape([
            x.shape[0], x.shape[1], 3, self.num_head,
            x.shape[2] // self.num_head
        ]).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv
        attn = q @ k.transpose(-1, -2) / math.sqrt(32 // self.num_head)
        y = attn @ v  # B H N h
        y = y.permute(0, 2, 1, 3).flatten(-2)
        return self.proj(y)


# models with dynamicop


def register_mutable(module: DynamicChannelMixin,
                     mutable: OneShotMutableChannelUnit,
                     is_out=True,
                     start=0,
                     end=-1):
    if end == -1:
        end = mutable.num_channels + start
    if is_out:
        container: MutableChannelContainer = module.get_mutable_attr(
            'out_channels')
    else:
        container: MutableChannelContainer = module.get_mutable_attr(
            'in_channels')
    container.register_mutable(mutable, start, end)


class SampleExpandDerivedMutable(BaseMutable):

    def __init__(self, expand_ratio=1) -> None:
        super().__init__()
        self.ratio = expand_ratio

    def __mul__(self, other):
        if isinstance(other, OneShotMutableChannel):

            def _expand_mask():
                mask = other.current_mask
                mask = torch.unsqueeze(
                    mask,
                    -1).expand(list(mask.shape) + [self.ratio]).flatten(-2)
                return mask

            return DerivedMutable(_expand_mask, _expand_mask, [self, other])
        else:
            raise NotImplementedError()

    def dump_chosen(self):
        return super().dump_chosen()

    def fix_chosen(self, chosen):
        return super().fix_chosen(chosen)

    def num_choices(self) -> int:
        return super().num_choices


class DynamicLinearModel(nn.Module):
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
            DynamicConv2d(3, 8, 3, 1, 1), DynamicBatchNorm2d(8), nn.ReLU(),
            DynamicConv2d(8, 16, 3, 1, 1), DynamicBatchNorm2d(16),
            nn.AdaptiveAvgPool2d(1))
        self.linear = DynamicLinear(16, 1000)

        MutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)
        self._register_mutable()

    def forward(self, x):
        x1 = self.net(x)
        x1 = x1.reshape([x1.shape[0], -1])
        return self.linear(x1)

    def _register_mutable(self):
        mutable1 = OneShotMutableChannel(8, candidate_choices=[1, 4, 8])
        mutable2 = OneShotMutableChannel(16, candidate_choices=[2, 8, 16])
        mutable_value = SampleExpandDerivedMutable(1)

        MutableChannelContainer.register_mutable_channel_to_module(
            self.net[0], mutable1, True)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.net[1], mutable1.expand_mutable_channel(1), True, 0, 8)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.net[3], mutable_value * mutable1, False, 0, 8)

        MutableChannelContainer.register_mutable_channel_to_module(
            self.net[3], mutable2, True)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.net[4], mutable2, True)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.linear, mutable2, False)
