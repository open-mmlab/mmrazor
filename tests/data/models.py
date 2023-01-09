# Copyright (c) OpenMMLab. All rights reserved.
# this file includes models for tesing.
from collections import OrderedDict
from typing import Dict
import math

from torch.nn import Module
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmengine.model import BaseModel
from mmrazor.models.architectures.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d, DynamicLinear, DynamicChannelMixin, DynamicPatchEmbed, DynamicSequential
from mmrazor.models.mutables.mutable_channel import MutableChannelContainer
from mmrazor.models.mutables import MutableChannelUnit
from mmrazor.models.mutables import DerivedMutable
from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutables import OneShotMutableChannelUnit, OneShotMutableChannel

from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.models.architectures.backbones.searchable_autoformer import TransformerEncoderLayer
from mmrazor.registry import MODELS
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.models.architectures.backbones.searchable_autoformer import TransformerEncoderLayer
from mmrazor.models.utils.parse_values import parse_values

from mmrazor.models.architectures.ops.mobilenet_series import MBBlock
from mmcv.cnn import ConvModule
from mmengine.model import Sequential
from mmrazor.models.architectures.utils.mutable_register import (
    mutate_conv_module, mutate_mobilenet_layer)

# models to test fx tracer


def untracable_function(x: torch.Tensor):
    if x.sum() > 0:
        x = x - 1
    else:
        x = x + 1
    return x


class UntracableModule(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if x.sum() > 0:
            x = x * 2
        else:
            x = x * -2
        x = self.conv2(x)
        return x


class ModuleWithUntracableMethod(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.untracable_method(x)
        x = self.conv2(x)
        return x

    def untracable_method(self, x):
        if x.sum() > 0:
            x = x * 2
        else:
            x = x * -2
        return x

@MODELS.register_module()
class UntracableBackBone(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 2)
        self.untracable_module = UntracableModule(16, 8)
        self.module_with_untracable_method = ModuleWithUntracableMethod(8, 16)

    def forward(self, x):
        x = self.conv(x)
        x = untracable_function(x)
        x = self.untracable_module(x)
        x = self.module_with_untracable_method(x)
        return x


class UntracableModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.backbone = UntracableBackBone()
        self.head = LinearHeadForTest(16, 1000)

    def forward(self, x):
        return self.head(self.backbone(x))



class ConvAttnModel(Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.head = LinearHeadForTest(16, 1000)

    def forward(self, x):
        x1 = self.conv(x)
        attn = F.sigmoid(self.pool(x1))
        x_attn = x1 * attn
        x_last = self.conv2(x_attn)
        return self.head(x_last)

@MODELS.register_module()
class LinearHeadForTest(Module):

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
        self.head = LinearHeadForTest(8, 1000)

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
        self.head = LinearHeadForTest(48, 1000)

    def forward(self, x):
        return self.head(self.net(x))


class SelfAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 4, 4, 4)

        self.num_head = 4
        self.qkv = nn.Linear(32, 32 * 3)
        self.proj = nn.Linear(32, 32)

        self.head = LinearHeadForTest(32, 1000)

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
                     mutable: MutableChannelUnit,
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

    def export_chosen(self):
        return super().export_chosen()

    def fix_chosen(self, chosen):
        return super().fix_chosen(chosen)

    def num_choices(self) -> int:
        return super().num_choices

    @property
    def current_choice(self):
        return super().current_choice

    @current_choice.setter
    def current_choice(self, choice):
        super().current_choice(choice)

    
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


class DynamicAttention(nn.Module):
    """
        x 
        |blocks: DynamicSequential(depth)
        |(blocks)
        x1
        |fc (OneShotMutableChannel * OneShotMutableValue)
        output
    """

    def __init__(self) -> None:
        super().__init__()

        self.mutable_depth = OneShotMutableValue(
            value_list=[1, 2], default_value=2)
        self.mutable_embed_dims = OneShotMutableChannel(
            num_channels=624, candidate_choices=[576, 624])
        self.base_embed_dims = OneShotMutableChannel(
            num_channels=64, candidate_choices=[64])
        self.mutable_num_heads = [
            OneShotMutableValue(value_list=[8, 10], default_value=10)
            for _ in range(2)
        ]
        self.mutable_mlp_ratios = [
            OneShotMutableValue(value_list=[3.0, 3.5, 4.0], default_value=4.0)
            for _ in range(2)
        ]
        self.mutable_q_embed_dims = [
            i * self.base_embed_dims for i in self.mutable_num_heads
        ]

        self.patch_embed = DynamicPatchEmbed(
            img_size=224,
            in_channels=3,
            embed_dims=self.mutable_embed_dims.num_channels)

        # cls token and pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 197, self.mutable_embed_dims.num_channels))
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.mutable_embed_dims.num_channels))

        layers = []
        for i in range(self.mutable_depth.max_choice):
            layer = TransformerEncoderLayer(
                embed_dims=self.mutable_embed_dims.num_channels,
                num_heads=self.mutable_num_heads[i].max_choice,
                mlp_ratio=self.mutable_mlp_ratios[i].max_choice)
            layers.append(layer)
        self.blocks = DynamicSequential(*layers)

        # OneShotMutableChannelUnit
        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        self.register_mutables()

    def register_mutables(self):
        # mutablevalue
        self.blocks.register_mutable_attr('depth', self.mutable_depth)
        # mutablechannel
        MutableChannelContainer.register_mutable_channel_to_module(
            self.patch_embed, self.mutable_embed_dims, True)

        for i in range(self.mutable_depth.max_choice):
            layer = self.blocks[i]
            layer.register_mutables(
                mutable_num_heads=self.mutable_num_heads[i],
                mutable_mlp_ratios=self.mutable_mlp_ratios[i],
                mutable_q_embed_dims=self.mutable_q_embed_dims[i],
                mutable_head_dims=self.base_embed_dims,
                mutable_embed_dims=self.mutable_embed_dims)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)
        embed_dims = self.mutable_embed_dims.current_choice
        cls_tokens = self.cls_token[..., :embed_dims].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[..., :embed_dims]
        x = self.blocks(x)
        return torch.mean(x[:, 1:], dim=1)


class DynamicMMBlock(nn.Module):

    arch_setting = dict(
        kernel_size=[  # [min_kernel_size, max_kernel_size, step]
            [3, 5, 2],
            [3, 5, 2],
            [3, 5, 2],
            [3, 5, 2],
            [3, 5, 2],
            [3, 5, 2],
            [3, 5, 2],
        ],
        num_blocks=[  # [min_num_blocks, max_num_blocks, step]
            [1, 2, 1],
            [3, 5, 1],
            [3, 6, 1],
            [3, 6, 1],
            [3, 8, 1],
            [3, 8, 1],
            [1, 2, 1],
        ],
        expand_ratio=[  # [min_expand_ratio, max_expand_ratio, step]
            [1, 1, 1],
            [4, 6, 1],
            [4, 6, 1],
            [4, 6, 1],
            [4, 6, 1],
            [6, 6, 1],
            [6, 6, 1]
        ],
        num_out_channels=[  # [min_channel, max_channel, step]
            [16, 24, 8],
            [24, 32, 8],
            [32, 40, 8],
            [64, 72, 8],
            [112, 128, 8],
            [192, 216, 8],
            [216, 224, 8]
        ])

    def __init__(
        self, 
        conv_cfg: Dict = dict(type='mmrazor.BigNasConv2d'),
        norm_cfg: Dict = dict(type='mmrazor.DynamicBatchNorm2d'),
    ) -> None:
        super().__init__()

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_list = ['Swish'] * 7
        self.stride_list = [1, 2, 2, 2, 1, 2, 1]
        self.with_se_list = [False, False, True, False, True, True, True]
        self.kernel_size_list = parse_values(self.arch_setting['kernel_size'])
        self.num_blocks_list = parse_values(self.arch_setting['num_blocks'])
        self.expand_ratio_list = \
            parse_values(self.arch_setting['expand_ratio'])
        self.num_channels_list = \
            parse_values(self.arch_setting['num_out_channels'])
        assert len(self.kernel_size_list) == len(self.num_blocks_list) == \
            len(self.expand_ratio_list) == len(self.num_channels_list)
        self.with_attentive_shortcut = True
        self.in_channels = 24

        self.first_out_channels_list = [16]
        self.first_conv = ConvModule(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='Swish'))

        self.layers = []
        for i, (num_blocks, kernel_sizes, expand_ratios, num_channels) in \
            enumerate(zip(self.num_blocks_list, self.kernel_size_list,
                          self.expand_ratio_list, self.num_channels_list)):
            inverted_res_layer = self._make_single_layer(
                out_channels=num_channels,
                num_blocks=num_blocks,
                kernel_sizes=kernel_sizes,
                expand_ratios=expand_ratios,
                act_cfg=self.act_list[i],
                stride=self.stride_list[i],
                use_se=self.with_se_list[i])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(inverted_res_layer)

        last_expand_channels = 1344
        self.out_channels = 1984
        self.last_out_channels_list = [1792, 1984]
        self.last_expand_ratio_list = [6]

        last_layers = Sequential(
            OrderedDict([('final_expand_layer',
                          ConvModule(
                              in_channels=self.in_channels,
                              out_channels=last_expand_channels,
                              kernel_size=1,
                              padding=0,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=dict(type='Swish'))),
                         ('pool', nn.AdaptiveAvgPool2d((1, 1))),
                         ('feature_mix_layer',
                          ConvModule(
                              in_channels=last_expand_channels,
                              out_channels=self.out_channels,
                              kernel_size=1,
                              padding=0,
                              bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=None,
                              act_cfg=dict(type='Swish')))]))
        self.add_module('last_conv', last_layers)
        self.layers.append(last_layers)
        
        self.register_mutables()

    def _make_single_layer(self, out_channels, num_blocks,
                           kernel_sizes, expand_ratios,
                           act_cfg, stride, use_se):
        _layers = []
        for i in range(max(num_blocks)):
            if i >= 1:
                stride = 1
            if use_se:
                se_cfg = dict(
                    act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')),
                    ratio=4,
                    conv_cfg=self.conv_cfg)
            else:
                se_cfg = None  # type: ignore

            mb_layer = MBBlock(
                in_channels=self.in_channels,
                out_channels=max(out_channels),
                kernel_size=max(kernel_sizes),
                stride=stride,
                expand_ratio=max(expand_ratios),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type=act_cfg),
                se_cfg=se_cfg,
                with_attentive_shortcut=self.with_attentive_shortcut)

            _layers.append(mb_layer)
            self.in_channels = max(out_channels)

        dynamic_seq = DynamicSequential(*_layers)
        return dynamic_seq

    def register_mutables(self):
        """Mutate the BigNAS-style MobileNetV3."""
        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        self.first_mutable_channels = OneShotMutableChannel(
            alias='backbone.first_channels',
            num_channels=max(self.first_out_channels_list),
            candidate_choices=self.first_out_channels_list)

        mutate_conv_module(
            self.first_conv, mutable_out_channels=self.first_mutable_channels)

        mid_mutable = self.first_mutable_channels
        # mutate the built mobilenet layers
        for i, layer in enumerate(self.layers[:-1]):
            num_blocks = self.num_blocks_list[i]
            kernel_sizes = self.kernel_size_list[i]
            expand_ratios = self.expand_ratio_list[i]
            out_channels = self.num_channels_list[i]

            prefix = 'backbone.layers.' + str(i + 1) + '.'

            mutable_out_channels = OneShotMutableChannel(
                alias=prefix + 'out_channels',
                candidate_choices=out_channels,
                num_channels=max(out_channels))

            mutable_kernel_size = OneShotMutableValue(
                alias=prefix + 'kernel_size', value_list=kernel_sizes)

            mutable_expand_ratio = OneShotMutableValue(
                alias=prefix + 'expand_ratio', value_list=expand_ratios)

            mutable_depth = OneShotMutableValue(
                alias=prefix + 'depth', value_list=num_blocks)
            layer.register_mutable_attr('depth', mutable_depth)

            for k in range(max(self.num_blocks_list[i])):
                mutate_mobilenet_layer(layer[k], mid_mutable,
                                       mutable_out_channels,
                                       mutable_expand_ratio,
                                       mutable_kernel_size)
                mid_mutable = mutable_out_channels

        self.last_mutable_channels = OneShotMutableChannel(
            alias='backbone.last_channels',
            num_channels=self.out_channels,
            candidate_choices=self.last_out_channels_list)

        last_mutable_expand_value = OneShotMutableValue(
            value_list=self.last_expand_ratio_list,
            default_value=max(self.last_expand_ratio_list))

        derived_expand_channels = mid_mutable * last_mutable_expand_value
        mutate_conv_module(
            self.layers[-1].final_expand_layer,
            mutable_in_channels=mid_mutable,
            mutable_out_channels=derived_expand_channels)
        mutate_conv_module(
            self.layers[-1].feature_mix_layer,
            mutable_in_channels=derived_expand_channels,
            mutable_out_channels=self.last_mutable_channels)

    def forward(self, x):
        x = self.first_conv(x)
        for _, layer in enumerate(self.layers):
            x = layer(x)

        return tuple([x])
