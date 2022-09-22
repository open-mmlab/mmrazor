# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from itertools import repeat
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd

from mmrazor.models.mutables.base_mutable import BaseMutable
from .dynamic_mixins import DynamicChannelMixin


def _ntuple(n: int) -> Callable:  # pragma: no cover
    """Repeat a number n times."""

    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def _get_current_kernel_pos(source_kernel_size: int,
                            target_kernel_size: int) -> Tuple[int, int]:
    """Get position of current kernel size.

    Returns:
        Tuple[int, int]: (upper left position, bottom right position)
    """
    assert source_kernel_size >= target_kernel_size, \
        '`source_kernel_size` must greater or equal than `target_kernel_size`'

    center = source_kernel_size >> 1
    current_offset = target_kernel_size >> 1

    start_offset = center - current_offset
    end_offset = center + current_offset + 1

    return start_offset, end_offset


def _get_same_padding(kernel_size: int, n_dims: int) -> Tuple[int]:
    """Get same padding according to kernel size."""
    assert kernel_size & 1
    _pair = _ntuple(n_dims)

    return _pair(kernel_size >> 1)


class DynamicConvMixin(DynamicChannelMixin):
    """A mixin class for Pytorch conv, which can mutate ``in_channels`` and
    ``out_channels``.

    Note:
        All subclass should implement ``conv_func``API.
    """

    @property
    @abstractmethod
    def conv_func(self: _ConvNd):
        """The function that will be used in ``forward_mixin``."""
        pass

    def register_mutable_attr(self, attr, mutable):

        if attr == 'in_channels':
            self._register_mutable_in_channels(mutable)
        elif attr == 'out_channels':
            self._register_mutable_out_channels(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_in_channels(
            self: _ConvNd, mutable_in_channels: BaseMutable) -> None:
        """Mutate ``in_channels`` with given mutable.

        Args:
            mutable_in_channels (BaseMutable): Mutable for controlling
                ``in_channels``.

        Raises:
            ValueError: Error if size of mask if not same as ``in_channels``.
        """
        assert hasattr(self, 'mutable_attrs')
        self.check_mutable_channels(mutable_in_channels)
        mask_size = mutable_in_channels.current_mask.size(0)
        if mask_size != self.in_channels:
            raise ValueError(
                f'Expect mask size of mutable to be {self.in_channels} as '
                f'`in_channels`, but got: {mask_size}.')

        self.mutable_attrs['in_channels'] = mutable_in_channels

    def _register_mutable_out_channels(
            self: _ConvNd, mutable_out_channels: BaseMutable) -> None:
        """Mutate ``out_channels`` with given mutable.

        Args:
            mutable_out_channels (BaseMutable): Mutable for controlling
                ``out_channels``.

        Raises:
            ValueError: Error if size of mask if not same as ``out_channels``.
        """
        assert hasattr(self, 'mutable_attrs')
        self.check_mutable_channels(mutable_out_channels)
        mask_size = mutable_out_channels.current_mask.size(0)
        if mask_size != self.out_channels:
            raise ValueError(
                f'Expect mask size of mutable to be {self.out_channels} as '
                f'`out_channels`, but got: {mask_size}.')

        self.mutable_attrs['out_channels'] = mutable_out_channels

    @property
    def mutable_in_channels(self: _ConvNd) -> Optional[BaseMutable]:
        """Mutable related to input."""
        assert hasattr(self, 'mutable_attrs')
        return getattr(self.mutable_attrs, 'in_channels', None)  # type:ignore

    @property
    def mutable_out_channels(self: _ConvNd) -> Optional[BaseMutable]:
        """Mutable related to output."""
        assert hasattr(self, 'mutable_attrs')
        return getattr(self.mutable_attrs, 'out_channels', None)  # type:ignore

    def get_dynamic_params(
            self: _ConvNd) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tuple[int]]: Sliced weight, bias
                and padding.
        """
        # slice in/out channel of weight according to
        # mutable in_channels/out_channels
        weight, bias = self._get_dynamic_params_by_mutable_channels(
            self.weight, self.bias)
        return weight, bias, self.padding

    def _get_dynamic_params_by_mutable_channels(
            self: _ConvNd, weight: Tensor,
            bias: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """Get sliced weight and bias according to ``mutable_in_channels`` and
        ``mutable_out_channels``.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Sliced weight and bias.
        """
        if 'in_channels' not in self.mutable_attrs and \
                'out_channels' not in self.mutable_attrs:
            return weight, bias

        if 'in_channels' in self.mutable_attrs:
            mutable_in_channels = self.mutable_attrs['in_channels']
            in_mask = mutable_in_channels.current_mask.to(weight.device)
        else:
            in_mask = torch.ones(weight.size(1)).bool().to(weight.device)

        if 'out_channels' in self.mutable_attrs:
            mutable_out_channels = self.mutable_attrs['out_channels']
            out_mask = mutable_out_channels.current_mask.to(weight.device)
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

    def forward_mixin(self: _ConvNd, x: Tensor) -> Tensor:
        """Forward of dynamic conv2d OP."""
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = x.size(1)
        weight, bias, padding = self.get_dynamic_params()

        return self.conv_func(x, weight, bias, self.stride, padding,
                              self.dilation, groups)

    def to_static_op(self: _ConvNd) -> nn.Conv2d:
        """Convert dynamic conv2d to :obj:`torch.nn.Conv2d`.

        Returns:
            torch.nn.Conv2d: :obj:`torch.nn.Conv2d` with sliced parameters.
        """
        self.check_if_mutables_fixed()

        weight, bias, padding = self.get_dynamic_params()
        groups = self.groups
        if groups == self.in_channels == self.out_channels and \
                self.mutable_in_channels is not None:
            mutable_in_channels = self.mutable_attrs['in_channels']
            groups = mutable_in_channels.current_mask.sum().item()
        out_channels = weight.size(0)
        in_channels = weight.size(1) * groups

        kernel_size = tuple(weight.shape[2:])

        static_conv = self.static_op_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            padding_mode=self.padding_mode,
            dilation=self.dilation,
            groups=groups,
            bias=True if bias is not None else False)

        static_conv.weight = nn.Parameter(weight)
        if bias is not None:
            static_conv.bias = nn.Parameter(bias)

        return static_conv


class BigNasConvMixin(DynamicConvMixin):
    """A mixin class for Pytorch conv, which can mutate ``in_channels``,
    ``out_channels`` and ``kernel_size``."""

    def register_mutable_attr(self, attr, mutable):

        if attr == 'in_channels':
            self._register_mutable_in_channels(mutable)
        elif attr == 'out_channels':
            self._register_mutable_out_channels(mutable)
        elif attr == 'kernel_size':
            self._register_mutable_kernel_size(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_kernel_size(
            self: _ConvNd, mutable_kernel_size: BaseMutable) -> None:
        """Mutate ``kernel_size`` with given mutable.

        Args:
            mutable_kernel_size (BaseMutable): Mutable for controlling
                ``kernel_size``.

        Note:
            ``kernel_size_seq`` must be provided if ``mutable_kernel_size``
            does not have ``choices`` attribute.

        Raises:
            ValueError: Error if max choice of ``kernel_size_list``
                not same as ``kernel_size``.
        """

        kernel_size_seq = getattr(mutable_kernel_size, 'choices', None)
        if kernel_size_seq is None or len(kernel_size_seq) == 0:
            raise ValueError('kernel size sequence must be provided')
        kernel_size_list = list(sorted(kernel_size_seq))

        _pair = _ntuple(len(self.weight.shape) - 2)
        max_kernel_size = _pair(kernel_size_list[-1])
        if max_kernel_size != self.kernel_size:
            raise ValueError(
                f'Expect max kernel size to be: {self.kernel_size}, '
                f'but got: {max_kernel_size}')

        self.kernel_size_list = kernel_size_list
        self.mutable_attrs['kernel_size'] = mutable_kernel_size

    def get_dynamic_params(
            self: _ConvNd) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tuple[int]]: Sliced weight, bias
                and padding.
        """
        # 1. slice kernel size of weight according to kernel size mutable
        weight, padding = self._get_dynamic_params_by_mutable_kernel_size(
            self.weight)

        # 2. slice in/out channel of weight according to mutable in_channels
        # and mutable out channels.
        weight, bias = self._get_dynamic_params_by_mutable_channels(
            weight, self.bias)
        return weight, bias, padding

    def _get_dynamic_params_by_mutable_kernel_size(
            self: _ConvNd, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        """Get sliced weight and bias according to ``mutable_in_channels`` and
        ``mutable_out_channels``."""
        if 'kernel_size' not in self.mutable_attrs \
                or self.kernel_size_list is None:
            return weight, self.padding

        mutable_kernel_size = self.mutable_attrs['kernel_size']
        current_kernel_size = self.get_current_choice(mutable_kernel_size)

        n_dims = len(self.weight.shape) - 2
        current_padding = _get_same_padding(current_kernel_size, n_dims)

        _pair = _ntuple(len(self.weight.shape) - 2)
        if _pair(current_kernel_size) == self.kernel_size:
            return weight, current_padding

        start_offset, end_offset = _get_current_kernel_pos(
            source_kernel_size=self.kernel_size[0],
            target_kernel_size=current_kernel_size)
        current_weight = \
            weight[:, :, start_offset:end_offset, start_offset:end_offset]

        return current_weight, current_padding

    def forward_mixin(self: _ConvNd, x: Tensor) -> Tensor:
        """Forward of dynamic conv2d OP."""
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = x.size(1)
        weight, bias, padding = self.get_dynamic_params()

        return self.conv_func(x, weight, bias, self.stride, padding,
                              self.dilation, groups)


class OFAConvMixin(BigNasConvMixin):
    """A mixin class for Pytorch conv, which can mutate ``in_channels``,
    ``out_channels`` and ``kernel_size``."""

    def _register_mutable_kernel_size(
            self: _ConvNd, mutable_kernel_size: BaseMutable) -> None:
        """Mutate ``kernel_size`` with given mutable and register
        transformation matrix."""
        super()._register_mutable_kernel_size(mutable_kernel_size)
        self._register_trans_matrix()

    def _register_trans_matrix(self: _ConvNd) -> None:
        """Register transformation matrix that used in progressive
        shrinking."""
        assert self.kernel_size_list is not None

        trans_matrix_names = []
        for i in range(len(self.kernel_size_list) - 1, 0, -1):
            source_kernel_size = self.kernel_size_list[i]
            target_kernel_size = self.kernel_size_list[i - 1]
            trans_matrix_name = self._get_trans_matrix_name(
                src=source_kernel_size, tar=target_kernel_size)
            trans_matrix_names.append(trans_matrix_name)
            # TODO support conv1d & conv3d
            trans_matrix = nn.Parameter(torch.eye(target_kernel_size**2))
            self.register_parameter(name=trans_matrix_name, param=trans_matrix)
        self._trans_matrix_names = trans_matrix_names

    @staticmethod
    def _get_trans_matrix_name(src: int, tar: int) -> str:
        """Get name of trans matrix."""
        return f'trans_matrix_{src}to{tar}'

    def _get_dynamic_params_by_mutable_kernel_size(
            self: _ConvNd, weight: Tensor) -> Tuple[Tensor, Tuple[int]]:
        """Get sliced weight and bias according to ``mutable_in_channels`` and
        ``mutable_out_channels``."""

        if 'kernel_size' not in self.mutable_attrs:
            return weight, self.padding

        mutable_kernel_size = self.mutable_attrs['kernel_size']
        current_kernel_size = self.get_current_choice(mutable_kernel_size)

        n_dims = len(self.weight.shape) - 2
        current_padding = _get_same_padding(current_kernel_size, n_dims)

        _pair = _ntuple(len(self.weight.shape) - 2)
        if _pair(current_kernel_size) == self.kernel_size:
            return weight, current_padding

        current_weight = weight[:, :, :, :]
        for i in range(len(self.kernel_size_list) - 1, 0, -1):
            source_kernel_size = self.kernel_size_list[i]
            if source_kernel_size <= current_kernel_size:
                break
            target_kernel_size = self.kernel_size_list[i - 1]
            trans_matrix = getattr(
                self,
                self._get_trans_matrix_name(
                    src=source_kernel_size, tar=target_kernel_size))

            start_offset, end_offset = _get_current_kernel_pos(
                source_kernel_size=source_kernel_size,
                target_kernel_size=target_kernel_size)
            target_weight = current_weight[:, :, start_offset:end_offset,
                                           start_offset:end_offset]
            target_weight = target_weight.reshape(-1, target_kernel_size**2)
            target_weight = F.linear(target_weight, trans_matrix)
            target_weight = target_weight.reshape(
                weight.size(0), weight.size(1), target_kernel_size,
                target_kernel_size)

            current_weight = target_weight

        return current_weight, current_padding
