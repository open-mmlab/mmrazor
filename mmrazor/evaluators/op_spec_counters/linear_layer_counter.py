# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import OP_SPEC_COUNTERS
from .base_counter import BaseCounter
from .flops_params_counter import get_model_parameters_number


@OP_SPEC_COUNTERS.register_module()
class LinearCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        input = input[0]
        output_last_dim = output.shape[
            -1]  # pytorch checks dimensions, so here we don't care much
        module.__flops__ += int(np.prod(input.shape) * output_last_dim)
        module.__params__ += get_model_parameters_number(module)
