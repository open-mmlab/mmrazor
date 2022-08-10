# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import OP_SPEC_COUNTERS
from .base_counter import BaseCounter
from .flops_params_counter import get_model_parameters_number


class BNCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        input = input[0]
        batch_flops = np.prod(input.shape)
        if getattr(module, 'affine', False):
            batch_flops *= 2
        module.__flops__ += int(batch_flops)
        module.__params__ += get_model_parameters_number(module)


@OP_SPEC_COUNTERS.register_module()
class BatchNorm1dCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class BatchNorm2dCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class BatchNorm3dCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class BatchNorm2dSliceCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class BatchNorm3dSliceCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class InstanceNorm1dCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class InstanceNorm2dCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class InstanceNorm3dCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class LayerNormCounter(BNCounter):
    pass


@OP_SPEC_COUNTERS.register_module()
class GroupNormCounter(BNCounter):
    pass
