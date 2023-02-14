# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_conv import \
    DynamicConv2d
from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_linear import \
    DynamicLinear


class GroupFisherMixin:
    """The mixin class for GroupFisher ops."""

    def _init(self) -> None:
        self.handlers: list = []
        self.recorded_input: List = []
        self.recorded_grad: List = []
        self.recorded_out_shape: List = []

    def forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: GroupFisherMixin, input, output):
            module.recorded_out_shape.append(output.shape)
            module.recorded_input.append(input[0])

        return forward_hook

    def backward_hook_wrapper(self):
        """Wrap the hook used in backward."""

        def backward_hook(module: GroupFisherMixin, grad_in, grad_out):
            module.recorded_grad.insert(0, grad_in[0])

        return backward_hook

    def start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.end_record()  # ensure to run start_record only once
        self.handlers.append(
            self.register_forward_hook(self.forward_hook_wrapper()))
        self.handlers.append(
            self.register_backward_hook(self.backward_hook_wrapper()))

    def end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def reset_recorded(self):
        """Reset the recorded information."""
        self.recorded_input = []
        self.recorded_grad = []
        self.recorded_out_shape = []

    @property
    def delta_flop_of_a_out_channel(self):
        raise NotImplementedError()

    @property
    def delta_flop_of_a_in_channel(self):
        raise NotImplementedError()

    @property
    def delta_memory_of_a_out_channel(self):
        raise NotImplementedError()


class GroupFisherConv2d(DynamicConv2d, GroupFisherMixin):
    """The Dynamic Conv2d operation used in GroupFisher Algorithm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init()

    @property
    def delta_flop_of_a_out_channel(self) -> torch.Tensor:
        """Calculate the summation of flops when prune an out_channel."""
        delta_flop_sum = 0
        for shape in self.recorded_out_shape:
            _, _, h, w = shape
            in_c = int(self.mutable_attrs['in_channels'].current_mask.float().
                       sum().item())
            # normal conv
            if self.groups == 1:
                delta_flop = h * w * self.kernel_size[0] * self.kernel_size[
                    1] * in_c
            # dwconv
            elif self.groups == self.in_channels == self.out_channels:
                delta_flop = h * w * self.kernel_size[0] * self.kernel_size[1]
            # groupwise conv
            else:
                raise NotImplementedError()
            delta_flop_sum += delta_flop
        return delta_flop_sum

    @property
    def delta_flop_of_a_in_channel(self):
        """Calculate the summation of flops when prune an in_channel."""
        delta_flop_sum = 0
        for shape in self.recorded_out_shape:
            _, out_c, h, w = shape
            # normal conv
            if self.groups == 1:
                delta_flop = h * w * self.kernel_size[0] * self.kernel_size[
                    1] * out_c
            # dwconv
            elif self.groups == self.in_channels == self.out_channels:
                delta_flop = h * w * self.kernel_size[0] * self.kernel_size[1]
            # groupwise conv
            else:
                raise NotImplementedError()
            delta_flop_sum += delta_flop
        return delta_flop_sum

    @property
    def delta_memory_of_a_out_channel(self):
        """Calculate the summation of memory when prune a channel."""
        delta_flop_sum = 0
        for shape in self.recorded_out_shape:
            _, _, h, w = shape
            delta_flop_sum += h * w
        return delta_flop_sum


class GroupFisherLinear(DynamicLinear, GroupFisherMixin):
    """The Dynamic Linear operation used in GroupFisher Algorithm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init()

    @property
    def delta_flop_of_a_out_channel(self):
        """Calculate the summation of flops when prune an out_channel."""
        in_c = self.mutable_attrs['in_channels'].current_mask.float().sum()
        return in_c * len(self.recorded_out_shape)

    @property
    def delta_flop_of_a_in_channel(self):
        """Calculate the summation of flops when prune an in_channel."""
        out_c = self.mutable_attrs['out_channels'].current_mask.float().sum()
        return out_c * len(self.recorded_out_shape)

    @property
    def delta_memory_of_a_out_channel(self):
        """Calculate the summation of memory when prune a channel."""
        return 1 * len(self.recorded_out_shape)
