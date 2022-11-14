# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import TASK_UTILS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


@TASK_UTILS.register_module()
class LinearCounter(BaseCounter):
    """FLOPs/params counter for Linear operation series."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        input = input[0]
        output_last_dim = output.shape[
            -1]  # pytorch checks dimensions, so here we don't care much
        module.__flops__ += int(np.prod(input.shape) * output_last_dim)
        module.__params__ += get_model_parameters_number(module)


@TASK_UTILS.register_module()
class DynamicLinearCounter(LinearCounter):
    pass
