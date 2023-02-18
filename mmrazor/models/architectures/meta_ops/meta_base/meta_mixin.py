from abc import abstractmethod
from itertools import repeat
from typing import Callable, Iterable, Optional, Tuple, Set

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd

from abc import ABC, abstractmethod

from mmrazor.models.mutables.base_mutable import BaseMutable



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




class MetaMixin(ABC):
    """Base class for dynamic OP. A dynamic OP usually consists of a normal
    static OP and mutables, where mutables are used to control the searchable
    (mutable) part of the dynamic OP.
    Note:
        When the dynamic OP has just been initialized, its forward propagation
        logic should be the same as the corresponding static OP. Only after
        the searchable part accepts the specific mutable through the
        corresponding interface does the part really become dynamic.
    Note:
        All subclass should implement ``to_static_op`` and
        ``static_op_factory`` APIs.
    Args:
        accepted_mutables (set): The string set of all accepted mutables.
    """
    accepted_mutable_attrs: Set[str] = set()
    attr_mappings: Dict[str, str] = dict()

    @abstractmethod
    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        pass

    def get_mutable_attr(self, attr: str) -> BaseMutable:

        self.check_mutable_attr_valid(attr)
        if attr in self.attr_mappings:
            attr_map = self.attr_mappings[attr]
            return getattr(self.mutable_attrs, attr_map, None)  # type:ignore
        else:
            return getattr(self.mutable_attrs, attr, None)  # type:ignore

    @classmethod
    @abstractmethod
    def convert_from(cls, module):
        """Convert an instance of Pytorch module to a new instance of Dynamic
        module."""

    @property
    @abstractmethod
    def static_op_factory(self):
        """Corresponding Pytorch OP."""

    @abstractmethod
    def to_static_op(self) -> nn.Module:
        """Convert dynamic OP to static OP.
        Note:
            The forward result for the same input between dynamic OP and its
            corresponding static OP must be same.
        Returns:
            nn.Module: Corresponding static OP.
        """

    def check_if_mutables_fixed(self):
        """Check if all mutables are fixed.
        Raises:
            RuntimeError: Error if a existing mutable is not fixed.
        """

        def check_fixed(mutable: Optional[BaseMutable]) -> None:
            if mutable is not None and not mutable.is_fixed:
                raise RuntimeError(f'Mutable {type(mutable)} is not fixed.')

        for mutable in self.mutable_attrs.values():  # type: ignore
            check_fixed(mutable)

    def check_mutable_attr_valid(self, attr):
        assert attr in self.attr_mappings or \
                    attr in self.accepted_mutable_attrs

    @staticmethod
    def get_current_choice(mutable: BaseMutable):
        """
        Get current choice of given mutable.
        Args:
            mutable (BaseMutable): Given mutable.
        Raises:
            RuntimeError: Error if `current_choice` is None.
        Returns:
            Any: Current choice of given mutable.
        """
        current_choice = mutable.current_choice
        if current_choice is None:
            raise RuntimeError(f'current choice of mutable {type(mutable)} '
                               'can not be None at runtime')

        return current_choice


class MetaConvMixin(DynamicChannelMixin):
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
            self: _ConvNd, mutable_in_channels: BaseMutable):
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
            self: _ConvNd, mutable_out_channels: BaseMutable):
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
    def mutable_in_channels(self: _ConvNd):
        """Mutable related to input."""
        assert hasattr(self, 'mutable_attrs')
        return getattr(self.mutable_attrs, 'in_channels', None)  # type:ignore

    @property
    def mutable_out_channels(self: _ConvNd):
        """Mutable related to output."""
        assert hasattr(self, 'mutable_attrs')
        return getattr(self.mutable_attrs, 'out_channels', None)  # type:ignore
    
    def forward_inpoup(self):
        if 'in_channels' in self.mutable_attrs:
            mutable_in_channels = self.mutable_attrs['in_channels']
            inp = mutable_in_channels.activated_channels
        if 'out_channels' in self.mutable_attrs:
            mutable_out_channels = self.mutable_attrs['out_channels']
            oup = mutable_out_channels.activated_channels
        return inp, oup

