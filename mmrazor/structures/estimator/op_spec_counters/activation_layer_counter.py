# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import OP_SPEC_COUNTERS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


@OP_SPEC_COUNTERS.register_module()
class ReLUCounter(BaseCounter):
    """FLOPs/params counter for ReLU series activate function."""

    @staticmethod
    def add_count_hook(module, input, output):
        active_elements_count = output.numel()
        module.__flops__ += int(active_elements_count)
        module.__params__ += get_model_parameters_number(module)


@OP_SPEC_COUNTERS.register_module()
class PReLUCounter(ReLUCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class ELUCounter(ReLUCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class LeakyReLUCounter(ReLUCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class ReLU6Counter(ReLUCounter):
    pass
