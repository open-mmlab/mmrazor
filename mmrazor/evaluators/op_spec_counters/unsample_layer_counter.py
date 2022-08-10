# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import OP_SPEC_COUNTERS
from .base_counter import BaseCounter
from .flops_params_counter import get_model_parameters_number


@OP_SPEC_COUNTERS.register_module()
class UnsampleCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        output_size = output[0]
        batch_size = output_size.shape[0]
        output_elements_count = batch_size
        for val in output_size.shape[1:]:
            output_elements_count *= val
        module.__flops__ += int(output_elements_count)
        module.__params__ += get_model_parameters_number(module)
