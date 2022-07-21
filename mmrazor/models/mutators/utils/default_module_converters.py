# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional

from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from ...architectures import (DynamicBatchNorm, DynamicConv2d,
                              DynamicGroupNorm, DynamicInstanceNorm,
                              DynamicLinear)


def dynamic_conv2d_converter(module: nn.Conv2d,
                             mutable_cfgs: Dict) -> DynamicConv2d:
    """Convert a nn.Conv2d module to a DynamicConv2d.

    Args:
        module (:obj:`torch.nn.Conv2d`): The original Conv2d module.
        in_channels_cfg (Dict): Config related to `in_channels`.
        out_channels_cfg (Dict): Config related to `out_channels`.
    """
    dynamic_conv = DynamicConv2d(
        mutable_cfgs=mutable_cfgs,
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


def dynamic_linear_converter(module: nn.Linear,
                             mutable_cfgs: Dict) -> DynamicLinear:
    """Convert a nn.Linear module to a DynamicLinear.

    Args:
        module (:obj:`torch.nn.Linear`): The original Linear module.
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """
    dynamic_linear = DynamicLinear(
        mutable_cfgs=mutable_cfgs,
        in_features=module.in_features,
        out_features=module.out_features,
        bias=True if module.bias is not None else False)
    return dynamic_linear


def dynamic_bn_converter(module: _BatchNorm,
                         mutable_cfgs: Dict) -> DynamicBatchNorm:
    """Convert a _BatchNorm module to a DynamicBatchNorm.

    Args:
        module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_bn = DynamicBatchNorm(
        mutable_cfgs=mutable_cfgs,
        num_features=module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)
    return dynamic_bn


def dynamic_in_converter(
        module: _InstanceNorm,
        in_channels_cfg: Dict,
        out_channels_cfg: Optional[Dict] = None) -> DynamicInstanceNorm:
    """Convert a _InstanceNorm module to a DynamicInstanceNorm.

    Args:
        module (:obj:`torch.nn._InstanceNorm`): The original InstanceNorm
            module.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_in = DynamicInstanceNorm(
        num_features_cfg=in_channels_cfg,
        num_features=module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)
    return dynamic_in


def dynamic_gn_converter(
        module: GroupNorm,
        in_channels_cfg: Dict,
        out_channels_cfg: Optional[Dict] = None) -> DynamicGroupNorm:
    """Convert a GroupNorm module to a DynamicGroupNorm.

    Args:
        module (:obj:`torch.nn.GroupNorm`): The original GroupNorm module.
        num_channels_cfg (Dict): Config related to `num_channels`.
    """
    dynamic_gn = DynamicGroupNorm(
        num_channels_cfg=in_channels_cfg,
        num_channels=module.num_channels,
        num_groups=module.num_groups,
        eps=module.eps,
        affine=module.affine)
    return dynamic_gn


DEFAULT_MODULE_CONVERTERS: Dict[Callable, Callable] = {
    nn.Conv2d: dynamic_conv2d_converter,
    nn.Linear: dynamic_linear_converter,
    nn.BatchNorm1d: dynamic_bn_converter,
    nn.BatchNorm2d: dynamic_bn_converter,
    nn.BatchNorm3d: dynamic_bn_converter,
    nn.InstanceNorm1d: dynamic_in_converter,
    nn.InstanceNorm2d: dynamic_in_converter,
    nn.InstanceNorm3d: dynamic_in_converter,
    nn.GroupNorm: dynamic_gn_converter
}
