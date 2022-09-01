# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import TASK_UTILS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


class BNCounter(BaseCounter):
    """FLOPs/params counter for BatchNormalization series."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        input = input[0]
        batch_flops = np.prod(input.shape)
        if getattr(module, 'affine', False):
            batch_flops *= 2
        module.__flops__ += int(batch_flops)
        module.__params__ += get_model_parameters_number(module)


@TASK_UTILS.register_module()
class BatchNorm1dCounter(BNCounter):
    """FLOPs/params counter for BatchNorm1d module."""
    pass


@TASK_UTILS.register_module()
class BatchNorm2dCounter(BNCounter):
    """FLOPs/params counter for BatchNorm2d module."""
    pass


@TASK_UTILS.register_module()
class BatchNorm3dCounter(BNCounter):
    """FLOPs/params counter for BatchNorm3d module."""
    pass


@TASK_UTILS.register_module()
class InstanceNorm1dCounter(BNCounter):
    """FLOPs/params counter for InstanceNorm1d module."""
    pass


@TASK_UTILS.register_module()
class InstanceNorm2dCounter(BNCounter):
    """FLOPs/params counter for InstanceNorm2d module."""
    pass


@TASK_UTILS.register_module()
class InstanceNorm3dCounter(BNCounter):
    """FLOPs/params counter for InstanceNorm3d module."""
    pass


@TASK_UTILS.register_module()
class LayerNormCounter(BNCounter):
    """FLOPs/params counter for LayerNorm module."""
    pass


@TASK_UTILS.register_module()
class GroupNormCounter(BNCounter):
    """FLOPs/params counter for GroupNorm module."""
    pass
