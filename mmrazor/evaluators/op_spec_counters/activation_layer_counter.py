# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import OP_SPEC_COUNTERS
from .base_counter import BaseCounter
from .flops_params_counter import get_model_parameters_number


@OP_SPEC_COUNTERS.register_module()
class ReLUCounter(BaseCounter):

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


@OP_SPEC_COUNTERS.register_module()
class PReLUSliceCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        active_elements_count = output.numel()
        module.__flops__ += int(active_elements_count)
        module.__params__ += get_model_parameters_number(module)
