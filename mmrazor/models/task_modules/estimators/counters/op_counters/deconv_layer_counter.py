# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import TASK_UTILS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


@TASK_UTILS.register_module()
class ConvTranspose2dCounter(BaseCounter):
    """FLOPs/params counter for Deconv module series."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Compute FLOPs and params based on the size of input & output."""
        # Can have multiple inputs, getting the first one
        input = input[0]

        batch_size = input.shape[0]
        input_height, input_width = input.shape[2:]

        # TODO: use more common representation
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = (
            kernel_height * kernel_width * in_channels * filters_per_channel)

        active_elements_count = batch_size * input_height * input_width
        overall_conv_flops = conv_per_position_flops * active_elements_count
        bias_flops = 0
        if module.bias is not None:
            output_height, output_width = output.shape[2:]
            bias_flops = out_channels * batch_size * output_height * output_height  # noqa: E501
        overall_flops = overall_conv_flops + bias_flops

        module.__flops__ += int(overall_flops)
        module.__params__ += get_model_parameters_number(module)
