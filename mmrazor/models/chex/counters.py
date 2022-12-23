# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn

from mmrazor.models.task_modules.estimators.counters import (Conv2dCounter,
                                                             LinearCounter)
from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DynamicConv2dCounter(Conv2dCounter):

    @staticmethod
    def add_count_hook(module: nn.Conv2d, input, output):

        input = input[0]

        batch_size = input.shape[0]
        output_dims = list(output.shape[2:])

        kernel_dims = list(module.kernel_size)

        out_channels = module.mutable_attrs['out_channels'].activated_channels
        in_channels = module.mutable_attrs['in_channels'].activated_channels

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
class DynamicLinearCounter(LinearCounter):
    pass


@TASK_UTILS.register_module()
class ChexLinearCounter(DynamicLinearCounter):
    pass


@TASK_UTILS.register_module()
class ChexConv2Counter(DynamicConv2dCounter):
    pass
