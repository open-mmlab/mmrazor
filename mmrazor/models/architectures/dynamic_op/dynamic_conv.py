# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections.abc import Iterable
from itertools import repeat
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.mutable_channel import MutableChannel
from mmrazor.models.mutables.mutable_value import OneShotMutableValue
from mmrazor.registry import MODELS
from .base import MUTABLE_CFGS_TYPE, ChannelDynamicOP


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


class DynamicConv2d(nn.Conv2d, ChannelDynamicOP):
    """Applies a 2D convolution over an input signal composed of several input
    planes according to the `mutable_in_channels` and `mutable_out_channels`
    dynamically.

    Args:
        in_channels_cfg (Dict): Config related to `in_channels`.
        out_channels_cfg (Dict): Config related to `out_channels`.
    """

    accepted_mutable_keys = {'in_channels', 'out_channels', 'kernel_size'}

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **conv_kwargs) -> None:
        super().__init__(**conv_kwargs)

        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_channels_mutable(mutable_cfgs)
        self._register_kernel_size_mutable(mutable_cfgs)

        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'

    def _register_channels_mutable(self,
                                   mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'in_channels' or 'out_channels' in mutable_cfgs:
            assert 'in_channels' in mutable_cfgs and \
                'out_channels' in mutable_cfgs, \
                'both `in_channels` and `out_channels` ' \
                'should be contained in `mutable_cfgs`'
            in_channels_cfg = copy.deepcopy(mutable_cfgs['in_channels'])
            in_channels_cfg.update(num_channels=self.in_channels)
            self.in_channels_mutable = MODELS.build(in_channels_cfg)

            out_channels_cfg = copy.deepcopy(mutable_cfgs['out_channels'])
            out_channels_cfg.update(dict(num_channels=self.out_channels))
            self.out_channels_mutable = MODELS.build(out_channels_cfg)

            assert isinstance(self.in_channels_mutable, MutableChannel)
            assert isinstance(self.out_channels_mutable, MutableChannel)
        else:
            self.register_parameter('in_channels_mutable', None)
            self.register_parameter('out_channels_mutable', None)

    def _register_kernel_size_mutable(self,
                                      mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'kernel_size' in mutable_cfgs:
            kernel_size_cfg = copy.deepcopy(mutable_cfgs['kernel_size'])
            self.kernel_size_mutable = MODELS.build(kernel_size_cfg)
            # FIXME
            # use correct type after MutableValue is implemented
            assert isinstance(self.kernel_size_mutable, OneShotMutableValue)
            self.kernel_size_mutable.current_choice = self.kernel_size[0]
        else:
            self.register_parameter('kernel_size_mutable', None)

    @property
    def mutable_in(self) -> MutableChannel:
        """Mutable `in_channels`."""
        return self.in_channels_mutable

    @property
    def mutable_out(self) -> MutableChannel:
        """Mutable `out_channels`."""
        return self.out_channels_mutable

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_channels` and
        `mutable_out_channels`, and forward."""
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias, padding = self._get_dynamic_params()

        return F.conv2d(input, weight, bias, self.stride, padding,
                        self.dilation, groups)

    def get_dynamic_params_by_kernel_size_mutable(
            self, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        assert self.kernel_size_mutable is None

        return weight, self.padding

    def get_dynamic_params_by_channels_mutable(
            self, weight: Tensor,
            bias: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        if self.in_channels_mutable is None:
            return weight, bias

        in_mask = self.in_channels_mutable.current_mask.to(weight.device)
        out_mask = self.out_channels_mutable.current_mask.to(weight.device)

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

    def _get_dynamic_params(
            self) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        # 1. slice kernel size of weight according to kernel size mutable
        weight, padding = self.get_dynamic_params_by_kernel_size_mutable(
            self.weight)

        # 2. slice in/out channel of weight according to in_channels/out_channels mutable  # noqa: E501
        weight, bias = self.get_dynamic_params_by_channels_mutable(
            weight, self.bias)
        return weight, bias, padding

    def to_static_op(self) -> nn.Conv2d:
        self.check_if_mutables_fixed()

        weight, bias, padding = self._get_dynamic_params()
        groups = self.groups
        if groups == self.in_channels == self.out_channels:
            groups = self.in_channels_mutable.current_mask.sum().item()
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


class ProgressiveDynamicConv2d(DynamicConv2d):

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **conv_kwargs) -> None:
        super().__init__(mutable_cfgs=mutable_cfgs, **conv_kwargs)

        self._kernel_size_list = self.kernel_size_mutable.choices.copy()
        max_choice = self.kernel_size_mutable.max_choice
        if _pair(max_choice) != self.kernel_size:
            raise ValueError('Max choice of kernel size mutable must be the '
                             'same as Conv2d kernel size, but got max '
                             f'choice: {max_choice}, expected max '
                             f'kernel size: {self.kernel_size[0]}.')
        # register transform matrix for progressive shrink
        self._register_transform_matrix()

    def _register_transform_matrix(self) -> None:
        assert self.kernel_size_mutable is not None

        transform_matrix_name_list = []
        for i in range(len(self._kernel_size_list) - 1, 0, -1):
            source_kernel_size = self._kernel_size_list[i]
            target_kernel_size = self._kernel_size_list[i - 1]
            transform_matrix_name = self._get_transform_matrix_name(
                src=source_kernel_size, tar=target_kernel_size)
            transform_matrix_name_list.append(transform_matrix_name)
            transform_matrix = nn.Parameter(torch.eye(target_kernel_size**2))
            self.register_parameter(
                name=transform_matrix_name, param=transform_matrix)
        self._transform_matrix_name_list = transform_matrix_name_list

    @staticmethod
    def _get_transform_matrix_name(src: int, tar: int) -> str:
        return f'transform_matrix_{src}to{tar}'

    def get_dynamic_params_by_kernel_size_mutable(
            self, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        current_kernel_size = self.kernel_size_mutable.current_choice
        current_padding = _get_same_padding(current_kernel_size)
        if _pair(current_kernel_size) == self.kernel_size:
            return weight, current_padding

        current_weight = weight[:, :, :, :]
        for i in range(len(self._kernel_size_list) - 1, 0, -1):
            source_kernel_size = self._kernel_size_list[i]
            if source_kernel_size <= current_kernel_size:
                break
            target_kernel_size = self._kernel_size_list[i - 1]
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


class CenterCropDynamicConv2d(DynamicConv2d):

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **conv_kwargs) -> None:
        super().__init__(mutable_cfgs=mutable_cfgs, **conv_kwargs)

        max_choice = self.kernel_size_mutable.max_choice
        if _pair(max_choice) != self.kernel_size:
            raise ValueError('Max choice of kernel size mutable must be the '
                             'same as Conv2d kernel size, but got max '
                             f'choice: {max_choice}, expected max '
                             f'kernel size: {self.kernel_size[0]}.')

        assert self.kernel_size_mutable is not None

    def get_dynamic_params_by_kernel_size_mutable(
            self, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        current_kernel_size = self.kernel_size_mutable.current_choice
        current_padding = _get_same_padding(current_kernel_size)
        if _pair(current_kernel_size) == self.kernel_size:
            return weight, current_padding

        start_offset, end_offset = _get_current_kernel_pos(
            source_kernel_size=self.kernel_size[0],
            target_kernel_size=current_kernel_size)
        current_weight = \
            weight[:, :, start_offset:end_offset, start_offset:end_offset]

        return current_weight, current_padding
