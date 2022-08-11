# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from itertools import repeat
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import CONV_LAYERS
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..base import ChannelDynamicOP


def _ntuple(n: int) -> Callable:

    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


def _get_current_kernel_pos(source_kernel_size: int,
                            target_kernel_size: int) -> Tuple[int, int]:
    assert source_kernel_size > target_kernel_size, \
        '`source_kernel_size` must greater than `target_kernel_size`'

    center = source_kernel_size >> 1
    current_offset = target_kernel_size >> 1

    start_offset = center - current_offset
    end_offset = center + current_offset + 1

    return start_offset, end_offset


def _get_same_padding(kernel_size: int) -> Tuple[int]:
    assert kernel_size & 1

    return _pair(kernel_size >> 1)


@CONV_LAYERS.register_module()
class DynamicConv2d(nn.Conv2d, ChannelDynamicOP):
    accpeted_mutables = {
        'mutable_in_channels', 'mutable_out_channels', 'mutable_kernel_size'
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'

        self.mutable_in_channels: Optional[BaseMutable] = None
        self.mutable_out_channels: Optional[BaseMutable] = None
        self.mutable_kernel_size: Optional[BaseMutable] = None
        self.kernel_size_list: Optional[List[int]] = None

    def mutate_in_channels(self, mutable_in_channels: BaseMutable) -> None:
        self.check_mutable_channels(mutable_in_channels)
        # TODO
        # add warning if self.mutable_in_channels is not None
        self.mutable_in_channels = mutable_in_channels

    def mutate_out_channels(self, mutable_out_channels: BaseMutable) -> None:
        self.check_mutable_channels(mutable_out_channels)
        # TODO
        # add warnings
        self.mutable_out_channels = mutable_out_channels

    def mutate_kernel_size(
            self,
            mutable_kernel_size: BaseMutable,
            kernel_size_seq: Optional[Sequence[int]] = None) -> None:
        if kernel_size_seq is None:
            kernel_size_seq = getattr(mutable_kernel_size, 'choices')
        if kernel_size_seq is None:
            raise ValueError('kernel size sequence must be provided')
        kernel_size_list = list(sorted(kernel_size_seq))
        max_kernel_size = _pair(kernel_size_list[-1])
        if max_kernel_size != self.kernel_size:
            raise ValueError(
                f'Expect max kernel size to be: {self.kernel_size}, '
                f'but got: {max_kernel_size}')

        self.kernel_size_list = kernel_size_list
        self.mutable_kernel_size = mutable_kernel_size

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable `in_channels`."""
        return self.mutable_in_channels

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable `out_channels`."""
        return self.mutable_out_channels

    def forward(self, input: Tensor) -> Tensor:
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias, padding = self._get_dynamic_params()

        return F.conv2d(input, weight, bias, self.stride, padding,
                        self.dilation, groups)

    def _get_dynamic_params(
            self) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        # 1. slice kernel size of weight according to kernel size mutable
        weight, padding = self.get_dynamic_params_by_mutable_kernel_size(
            self.weight)

        # 2. slice in/out channel of weight according to in_channels/out_channels mutable  # noqa: E501
        weight, bias = self.get_dynamic_params_by_mutable_channels(
            weight, self.bias)
        return weight, bias, padding

    def get_dynamic_params_by_mutable_kernel_size(
            self, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        return weight, self.padding

    def get_dynamic_params_by_mutable_channels(
            self, weight: Tensor,
            bias: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        if self.mutable_in_channels is None and \
                self.mutable_out_channels is None:
            return weight, bias

        if self.mutable_in_channels is not None:
            in_mask = self.mutable_in_channels.current_mask.to(weight.device)
        else:
            in_mask = torch.ones(weight.size(1)).bool().to(weight.device)
        if self.mutable_out_channels is not None:
            out_mask = self.mutable_out_channels.current_mask.to(weight.device)
        else:
            out_mask = torch.ones(weight.size(0)).bool().to(weight.device)

        if self.groups == 1:
            weight = weight[out_mask][:, in_mask]
        elif self.groups == self.in_channels == self.out_channels:
            # depth-wise conv
            weight = weight[out_mask]
        else:
            raise NotImplementedError(
                'Current `ChannelMutator` only support pruning the depth-wise '
                '`nn.Conv2d` or `nn.Conv2d` module whose group number equals '
                f'to one, but got {self.groups}.')
        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def to_static_op(self) -> nn.Conv2d:
        self.check_if_mutables_fixed()

        weight, bias, padding = self._get_dynamic_params()
        groups = self.groups
        if groups == self.in_channels == self.out_channels and \
                self.mutable_in_channels is not None:
            groups = self.mutable_in_channels.current_mask.sum().item()
        out_channels = weight.size(0)
        in_channels = weight.size(1) * groups

        kernel_size = tuple(weight.shape[2:])

        static_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            padding_mode=self.padding_mode,
            dilation=self.dilation,
            groups=groups,
            bias=True if bias is not None else False)

        static_conv2d.weight = nn.Parameter(weight)
        if bias is not None:
            static_conv2d.bias = nn.Parameter(bias)

        return static_conv2d


@CONV_LAYERS.register_module()
class ProgressiveDynamicConv2d(DynamicConv2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _register_transform_matrix(self) -> None:
        assert self.kernel_size_list is not None

        transform_matrix_name_list = []
        for i in range(len(self.kernel_size_list) - 1, 0, -1):
            source_kernel_size = self.kernel_size_list[i]
            target_kernel_size = self.kernel_size_list[i - 1]
            transform_matrix_name = self._get_transform_matrix_name(
                src=source_kernel_size, tar=target_kernel_size)
            transform_matrix_name_list.append(transform_matrix_name)
            transform_matrix = nn.Parameter(torch.eye(target_kernel_size**2))
            self.register_parameter(
                name=transform_matrix_name, param=transform_matrix)
        self._transform_matrix_name_list = transform_matrix_name_list

    def mutate_kernel_size(
            self,
            mutable_kernel_size: BaseMutable,
            kernel_size_seq: Optional[Sequence[int]] = None) -> None:
        super().mutate_kernel_size(mutable_kernel_size, kernel_size_seq)

        self._register_transform_matrix()

    @staticmethod
    def _get_transform_matrix_name(src: int, tar: int) -> str:
        return f'transform_matrix_{src}to{tar}'

    def get_dynamic_params_by_mutable_kernel_size(
            self, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        if self.mutable_kernel_size is None or self.kernel_size_list is None:
            return weight, self.padding

        current_kernel_size = self.get_current_choice(self.mutable_kernel_size)
        current_padding = _get_same_padding(current_kernel_size)
        if _pair(current_kernel_size) == self.kernel_size:
            return weight, current_padding

        current_weight = weight[:, :, :, :]
        for i in range(len(self.kernel_size_list) - 1, 0, -1):
            source_kernel_size = self.kernel_size_list[i]
            if source_kernel_size <= current_kernel_size:
                break
            target_kernel_size = self.kernel_size_list[i - 1]
            transform_matrix = getattr(
                self,
                self._get_transform_matrix_name(
                    src=source_kernel_size, tar=target_kernel_size))

            start_offset, end_offset = _get_current_kernel_pos(
                source_kernel_size=source_kernel_size,
                target_kernel_size=target_kernel_size)
            target_weight = current_weight[:, :, start_offset:end_offset,
                                           start_offset:end_offset]
            target_weight = target_weight.reshape(-1, target_kernel_size**2)
            target_weight = F.linear(target_weight, transform_matrix)
            target_weight = target_weight.reshape(
                weight.size(0), weight.size(1), target_kernel_size,
                target_kernel_size)

            current_weight = target_weight

        return current_weight, current_padding


@CONV_LAYERS.register_module()
class CenterCropDynamicConv2d(DynamicConv2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_dynamic_params_by_mutable_kernel_size(
            self, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        if self.mutable_kernel_size is None or self.kernel_size_list is None:
            return weight, self.padding

        current_kernel_size = self.get_current_choice(self.mutable_kernel_size)
        current_padding = _get_same_padding(current_kernel_size)
        if _pair(current_kernel_size) == self.kernel_size:
            return weight, current_padding

        start_offset, end_offset = _get_current_kernel_pos(
            source_kernel_size=self.kernel_size[0],
            target_kernel_size=current_kernel_size)
        current_weight = \
            weight[:, :, start_offset:end_offset, start_offset:end_offset]

        return current_weight, current_padding
