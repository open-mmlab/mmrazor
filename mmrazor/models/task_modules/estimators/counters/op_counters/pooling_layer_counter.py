# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import TASK_UTILS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


class PoolCounter(BaseCounter):
    """FLOPs/params counter for Pooling series."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        input = input[0]
        module.__flops__ += int(np.prod(input.shape))
        module.__params__ += get_model_parameters_number(module)


@TASK_UTILS.register_module()
class MaxPool1dCounter(PoolCounter):
    """FLOPs/params counter for MaxPool1d module."""
    pass


@TASK_UTILS.register_module()
class MaxPool2dCounter(PoolCounter):
    """FLOPs/params counter for MaxPool2d module."""
    pass


@TASK_UTILS.register_module()
class MaxPool3dCounter(PoolCounter):
    """FLOPs/params counter for MaxPool3d module."""
    pass


@TASK_UTILS.register_module()
class AvgPool1dCounter(PoolCounter):
    """FLOPs/params counter for AvgPool1d module."""
    pass


@TASK_UTILS.register_module()
class AvgPool2dCounter(PoolCounter):
    """FLOPs/params counter for AvgPool2d module."""
    pass


@TASK_UTILS.register_module()
class AvgPool3dCounter(PoolCounter):
    """FLOPs/params counter for AvgPool3d module."""
    pass


@TASK_UTILS.register_module()
class AdaptiveMaxPool1dCounter(PoolCounter):
    """FLOPs/params counter for AdaptiveMaxPool1d module."""
    pass


@TASK_UTILS.register_module()
class AdaptiveMaxPool2dCounter(PoolCounter):
    """FLOPs/params counter for AdaptiveMaxPool2d module."""
    pass


@TASK_UTILS.register_module()
class AdaptiveMaxPool3dCounter(PoolCounter):
    """FLOPs/params counter for AdaptiveMaxPool3d module."""
    pass


@TASK_UTILS.register_module()
class AdaptiveAvgPool1dCounter(PoolCounter):
    """FLOPs/params counter for AdaptiveAvgPool1d module."""
    pass


@TASK_UTILS.register_module()
class AdaptiveAvgPool2dCounter(PoolCounter):
    """FLOPs/params counter for AdaptiveAvgPool2d module."""
    pass


@TASK_UTILS.register_module()
class AdaptiveAvgPool3dCounter(PoolCounter):
    """FLOPs/params counter for AdaptiveAvgPool3d module."""
    pass
