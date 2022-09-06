# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ...architectures.dynamic_op.bricks import (DynamicBatchNorm1d,
                                                DynamicBatchNorm2d,
                                                DynamicBatchNorm3d,
                                                DynamicConv2d, DynamicLinear)


def dynamic_conv2d_converter(module: nn.Conv2d) -> DynamicConv2d:
    """Convert a nn.Conv2d module to a DynamicConv2d.

    Args:
        module (:obj:`torch.nn.Conv2d`): The original Conv2d module.
        in_channels_cfg (Dict): Config related to `in_channels`.
        out_channels_cfg (Dict): Config related to `out_channels`.
    """
    dynamic_conv = DynamicConv2d.convert_from(module)
    return dynamic_conv


def dynamic_linear_converter(module: nn.Linear) -> DynamicLinear:
    """Convert a nn.Linear module to a DynamicLinear.

    Args:
        module (:obj:`torch.nn.Linear`): The original Linear module.
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """
    dynamic_linear = DynamicLinear.convert_from(module)
    return dynamic_linear


def dynamic_batch_norm_1d_converter(module: _BatchNorm) -> DynamicBatchNorm1d:
    """Convert a _BatchNorm module to a DynamicBatchNorm1d.

    Args:
        module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_bn = DynamicBatchNorm1d.convert_from(module)
    return dynamic_bn


def dynamic_batch_norm_2d_converter(module: _BatchNorm) -> DynamicBatchNorm2d:
    """Convert a _BatchNorm module to a DynamicBatchNorm2d.

    Args:
        module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_bn = DynamicBatchNorm2d.convert_from(module)
    return dynamic_bn


def dynamic_batch_norm_3d_converter(module: _BatchNorm) -> DynamicBatchNorm3d:
    """Convert a _BatchNorm module to a DynamicBatchNorm3d.

    Args:
        module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        num_features_cfg (Dict): Config related to `num_features`.
    """
    dynamic_bn = DynamicBatchNorm3d.convert_from(module)
    return dynamic_bn


DEFAULT_MODULE_CONVERTERS: Dict[Callable, Callable] = {
    nn.Conv2d: dynamic_conv2d_converter,
    nn.Linear: dynamic_linear_converter,
    nn.BatchNorm1d: dynamic_batch_norm_1d_converter,
    nn.BatchNorm2d: dynamic_batch_norm_2d_converter,
    nn.BatchNorm3d: dynamic_batch_norm_3d_converter
}
