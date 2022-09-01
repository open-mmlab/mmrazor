# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import TASK_UTILS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


@TASK_UTILS.register_module()
class ReLUCounter(BaseCounter):
    """FLOPs/params counter for ReLU series activate function."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        active_elements_count = output.numel()
        module.__flops__ += int(active_elements_count)
        module.__params__ += get_model_parameters_number(module)


@TASK_UTILS.register_module()
class PReLUCounter(ReLUCounter):
    """FLOPs/params counter for PReLU function."""
    pass


@TASK_UTILS.register_module()
class ELUCounter(ReLUCounter):
    """FLOPs/params counter for ELU function."""
    pass


@TASK_UTILS.register_module()
class LeakyReLUCounter(ReLUCounter):
    """FLOPs/params counter for LeakyReLU function."""
    pass


@TASK_UTILS.register_module()
class ReLU6Counter(ReLUCounter):
    """FLOPs/params counter for ReLU6 function."""
    pass
