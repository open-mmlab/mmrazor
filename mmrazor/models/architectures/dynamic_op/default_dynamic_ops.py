# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from mmrazor.models.mutables import MutableManageMixIn
from mmrazor.registry import MODELS


class DynamicConv2d(nn.Conv2d, MutableManageMixIn):
    """Applies a 2D convolution over an input signal composed of several input
    planes according to the `mutable_in_channels` and `mutable_out_channels`
    dynamically.

    Args:
        module_name (str): Name of this `DynamicConv2d`.
        in_channels_cfg (Dict): Config related to `in_channels`.
        out_channels_cfg (Dict): Config related to `out_channels`.
    """

    def __init__(self, module_name, in_channels_cfg, out_channels_cfg, *args,
                 **kwargs):
        super(DynamicConv2d, self).__init__(*args, **kwargs)

        in_channels_cfg = copy.deepcopy(in_channels_cfg)
        in_channels_cfg.update(
            dict(
                name=module_name,
                num_channels=self.in_channels,
                mask_type='in_mask'))
        self.mutable_in_channels = MODELS.build(in_channels_cfg)

        out_channels_cfg = copy.deepcopy(out_channels_cfg)
        out_channels_cfg.update(
            dict(
                name=module_name,
                num_channels=self.out_channels,
                mask_type='out_mask'))
        self.mutable_out_channels = MODELS.build(out_channels_cfg)

    @property
    def mutable_in(self):
        """Mutable `in_channels`."""
        return self.mutable_in_channels

    @property
    def mutable_out(self):
        """Mutable `out_channels`."""
        return self.mutable_out_channels

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_channels` and
        `mutable_out_channels`, and forward."""
        in_mask = self.mutable_in_channels.mask
        out_mask = self.mutable_out_channels.mask

        if self.groups == 1:
            weight = self.weight[out_mask][:, in_mask]
            groups = 1
        elif self.groups == self.in_channels == self.out_channels:
            # depth-wise conv
            weight = self.weight[out_mask]
            groups = input.size(1)
        else:
            raise NotImplementedError(
                'Current `ChannelMutator` only support pruning the depth-wise '
                '`nn.Conv2d` or `nn.Conv2d` module whose group number equals '
                'to one, but got {self.groups}.')

        bias = self.bias[out_mask] if self.bias is not None else None

        return F.conv2d(input, weight, bias, self.stride, self.padding,
                        self.dilation, groups)


class DynamicLinear(nn.Linear, MutableManageMixIn):
    """Applies a linear transformation to the incoming data according to the
    `mutable_in_features` and `mutable_out_features` dynamically.

    Args:
        module_name (str): Name of this `DynamicLinear`.
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """

    def __init__(self, module_name, in_features_cfg, out_features_cfg, *args,
                 **kwargs):
        super(DynamicLinear, self).__init__(*args, **kwargs)

        in_features_cfg = copy.deepcopy(in_features_cfg)
        in_features_cfg.update(
            dict(
                name=module_name,
                num_channels=self.in_features,
                mask_type='in_mask'))
        self.mutable_in_features = MODELS.build(in_features_cfg)

        out_features_cfg = copy.deepcopy(out_features_cfg)
        out_features_cfg.update(
            dict(
                name=module_name,
                num_channels=self.out_features,
                mask_type='out_mask'))
        self.mutable_out_features = MODELS.build(out_features_cfg)

    @property
    def mutable_in(self):
        """Mutable `in_features`."""
        return self.mutable_in_features

    @property
    def mutable_out(self):
        """Mutable `out_features`."""
        return self.mutable_out_features

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_features` and
        `mutable_out_features`, and forward."""
        in_mask = self.mutable_in_features.mask
        out_mask = self.mutable_out_features.mask

        weight = self.weight[out_mask][:, in_mask]
        bias = self.bias[out_mask] if self.bias is not None else None

        return F.linear(input, weight, bias)


class DynamicBatchNorm(_BatchNorm, MutableManageMixIn):
    """Applies Batch Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        module_name (str): Name of this `DynamicBatchNorm`.
        num_features_cfg (Dict): Config related to `num_features`.
    """

    def __init__(self, module_name, num_features_cfg, *args, **kwargs):
        super(DynamicBatchNorm, self).__init__(*args, **kwargs)

        num_features_cfg = copy.deepcopy(num_features_cfg)
        num_features_cfg.update(
            dict(
                name=module_name,
                num_channels=self.num_features,
                mask_type='out_mask'))
        self.mutable_num_features = MODELS.build(num_features_cfg)

    @property
    def mutable_in(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = \
                    self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)

        if self.affine:
            out_mask = self.mutable_num_features.mask
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            out_mask = self.mutable_num_features.mask
            running_mean = self.running_mean[out_mask] \
                if not self.training or self.track_running_stats else None
            running_var = self.running_var[out_mask] \
                if not self.training or self.track_running_stats else None
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return F.batch_norm(input, running_mean, running_var, weight, bias,
                            bn_training, exponential_average_factor, self.eps)


class DynamicInstanceNorm(_InstanceNorm, MutableManageMixIn):
    """Applies Instance Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        module_name (str): Name of this `DynamicInstanceNorm`.
        num_features_cfg (Dict): Config related to `num_features`.
    """

    def __init__(self, module_name, num_features_cfg, *args, **kwargs):
        super(DynamicInstanceNorm, self).__init__(*args, **kwargs)

        num_features_cfg = copy.deepcopy(num_features_cfg)
        num_features_cfg.update(
            dict(
                name=module_name,
                num_channels=self.num_features,
                mask_type='out_mask'))
        self.mutable_num_features = MODELS.build(num_features_cfg)

    @property
    def mutable_in(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_features.mask
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            out_mask = self.mutable_num_features.mask
            running_mean = self.running_mean[out_mask]
            running_var = self.running_var[out_mask]
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return F.instance_norm(input, running_mean, running_var, weight, bias,
                               self.training or not self.track_running_stats,
                               self.momentum, self.eps)


class DynamicGroupNorm(GroupNorm, MutableManageMixIn):
    """Applies Group Normalization over a mini-batch of inputs according to the
    `mutable_num_channels` dynamically.

    Args:
        module_name (str): Name of this `DynamicGroupNorm`.
        num_channels_cfg (Dict): Config related to `num_channels`.
    """

    def __init__(self, module_name, num_channels_cfg, *args, **kwargs):
        super(DynamicGroupNorm, self).__init__(*args, **kwargs)

        num_channels_cfg = copy.deepcopy(num_channels_cfg)
        num_channels_cfg.update(
            dict(
                name=module_name,
                num_channels=self.num_channels,
                mask_type='out_mask'))
        self.mutable_num_channels = MODELS.build(num_channels_cfg)

    @property
    def mutable_in(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    @property
    def mutable_out(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_channels`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_channels.mask
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        return F.group_norm(input, self.num_groups, weight, bias, self.eps)


def build_dynamic_conv2d(module: nn.Conv2d, module_name: str,
                         in_channels_cfg: Dict,
                         out_channels_cfg: Dict) -> DynamicConv2d:
    """Build DynamicConv2d.

    Args:
        module (:obj:`torch.nn.Conv2d`): The original Conv2d module.
        module_name (str): Name of this `DynamicConv2d`.
        in_channels_cfg (Dict): Config related to `in_channels`.
        out_channels_cfg (Dict): Config related to `out_channels`.
    """
    dynamic_conv = DynamicConv2d(
        module_name=module_name,
        in_channels_cfg=in_channels_cfg,
        out_channels_cfg=out_channels_cfg,
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=True if module.bias is not None else False,
        padding_mode=module.padding_mode)
    return dynamic_conv


def build_dynamic_linear(module: nn.Linear, module_name: str,
                         in_features_cfg: Dict,
                         out_features_cfg: Dict) -> DynamicLinear:
    """Build DynamicLinear.

    Args:
        module (:obj:`torch.nn.Linear`): The original Linear module.
        module_name (str): Name of this `DynamicLinear`.
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """
    dynamic_linear = DynamicLinear(
        module_name=module_name,
        in_features_cfg=in_features_cfg,
        out_features_cfg=out_features_cfg,
        in_features=module.in_features,
        out_features=module.out_features,
        bias=True if module.bias is not None else False)
    return dynamic_linear


def build_dynamic_bn(module: _BatchNorm, module_name: str,
                     num_features_cfg: Dict) -> DynamicBatchNorm:
    """Build DynamicBatchNorm.

    Args:
        module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        module_name (str): Name of this `DynamicBatchNorm`.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_bn = DynamicBatchNorm(
        module_name=module_name,
        num_features_cfg=num_features_cfg,
        num_features=module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)
    return dynamic_bn


def build_dynamic_in(module: _InstanceNorm, module_name: str,
                     num_features_cfg: Dict) -> DynamicInstanceNorm:
    """Build DynamicInstanceNorm.

    Args:
        module (:obj:`torch.nn._InstanceNorm`): The original InstanceNorm
            module.
        module_name (str): Name of this `DynamicInstanceNorm`.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_in = DynamicInstanceNorm(
        module_name=module_name,
        num_features_cfg=num_features_cfg,
        num_features=module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)
    return dynamic_in


def build_dynamic_gn(module: GroupNorm, module_name: str,
                     num_channels_cfg: Dict) -> DynamicGroupNorm:
    """Build DynamicGroupNorm.

    Args:
        module (:obj:`torch.nn.GroupNorm`): The original GroupNorm module.
        module_name (str): Name of this `DynamicGroupNorm`.
        num_channels_cfg (Dict): Config related to `num_channels`.
    """
    dynamic_gn = DynamicGroupNorm(
        module_name=module_name,
        num_channels_cfg=num_channels_cfg,
        num_channels=module.num_channels,
        num_groups=module.num_groups,
        eps=module.eps,
        affine=module.affine)
    return dynamic_gn
