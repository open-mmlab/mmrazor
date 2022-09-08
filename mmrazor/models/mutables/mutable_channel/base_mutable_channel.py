# Copyright (c) OpenMMLab. All rights reserved.
""""""
from abc import abstractproperty

import torch

from ..base_mutable import BaseMutable
from ..derived_mutable import DerivedMethodMixin


class BaseMutableChannel(BaseMutable, DerivedMethodMixin):
    """BaseMutableChannel works as a channel mask for DynamicOps to select
    channels.

    |---------------------------------------|
    |mutable_in_channel(BaseMutableChannel) |
    |---------------------------------------|
    |             DynamicOp                 |
    |---------------------------------------|
    |mutable_out_channel(BaseMutableChannel)|
    |---------------------------------------|

    Important interfaces:
        current_choice: used to get/set mask.
        current_mask: get mask(used in DynamicOps to get mask).
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.name = ''
        self.num_channels = num_channels

    # choice

    @abstractproperty
    def current_choice(self):
        """get current choice."""
        raise NotImplementedError()

    @current_choice.setter
    def current_choice(self):
        """set current choice."""
        raise NotImplementedError()

    @abstractproperty
    def current_mask(self) -> torch.Tensor:
        """Return a mask indicating the channel selection."""
        raise NotImplementedError()

    @property
    def activated_channels(self) -> int:
        """Number of activated channels."""
        return (self.current_mask == 1).sum().item()

    # implementation of abstract methods

    def fix_chosen(self, chosen=None):
        """Fix the mutable  with chosen."""
        if chosen is not None:
            self.current_choice = chosen

        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        self.is_fixed = True

    def dump_chosen(self):
        """dump current choice to a dict."""
        raise NotImplementedError()

    def num_choices(self) -> int:
        """Number of available choices."""
        raise NotImplementedError()

    # others

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'num_channels={self.num_channels}, '
        return repr_str
