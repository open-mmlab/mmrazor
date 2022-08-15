# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import OP_SPEC_COUNTERS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


class PoolCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        input = input[0]
        module.__flops__ += int(np.prod(input.shape))
        module.__params__ += get_model_parameters_number(module)


@OP_SPEC_COUNTERS.register_module()
class MaxPool1dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class MaxPool2dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class MaxPool3dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AvgPool1dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AvgPool2dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AvgPool3dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AdaptiveMaxPool1dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AdaptiveMaxPool2dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AdaptiveMaxPool3dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AdaptiveAvgPool1dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AdaptiveAvgPool2dCounter(PoolCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class AdaptiveAvgPool3dCounter(PoolCounter):
    pass
