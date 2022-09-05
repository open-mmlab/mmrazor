# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import TASK_UTILS
from .base_counter import BaseCounter


class ConvCounter(BaseCounter):
    """FLOPs/params counter for Conv module series."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        # Can have multiple inputs, getting the first one
        input = input[0]

        batch_size = input.shape[0]
        output_dims = list(output.shape[2:])

        kernel_dims = list(module.kernel_size)
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        filters_per_channel = out_channels / groups
        conv_per_position_flops = int(
            np.prod(kernel_dims)) * in_channels * filters_per_channel

        active_elements_count = batch_size * int(np.prod(output_dims))

        overall_conv_flops = conv_per_position_flops * active_elements_count
        overall_params = conv_per_position_flops

        bias_flops = 0
        overall_params = conv_per_position_flops
        if module.bias is not None:
            bias_flops = out_channels * active_elements_count
            overall_params += out_channels

        overall_flops = overall_conv_flops + bias_flops

        module.__flops__ += overall_flops
        module.__params__ += int(overall_params)


@TASK_UTILS.register_module()
class Conv1dCounter(ConvCounter):
    """FLOPs/params counter for Conv1d module."""
    pass


@TASK_UTILS.register_module()
class Conv2dCounter(ConvCounter):
    """FLOPs/params counter for Conv2d module."""
    pass


@TASK_UTILS.register_module()
class Conv3dCounter(ConvCounter):
    """FLOPs/params counter for Conv3d module."""
    pass
