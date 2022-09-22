# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

import torch

from mmrazor.registry import MODELS
from ..derived_mutable import DerivedMutable
from .base_mutable_channel import BaseMutableChannel

# TODO discuss later


@MODELS.register_module()
class SquentialMutableChannel(BaseMutableChannel):
    """SquentialMutableChannel defines a BaseMutableChannel which switch off
    channel mask from right to left sequentially, like '11111000'.

    A choice of SquentialMutableChannel is an integer, which indicates how many
    channel are activated from left to right.

    Args:
        num_channels (int): number of channels.
    """

    def __init__(self, num_channels: int, **kwargs):

        super().__init__(num_channels, **kwargs)
        self.mask = torch.ones([self.num_channels]).bool()

    @property
    def current_choice(self) -> int:
        """Get current choice."""
        return (self.mask == 1).sum().item()

    @current_choice.setter
    def current_choice(self, choice: int):
        """Set choice."""
        mask = torch.zeros([self.num_channels], device=self.mask.device)
        mask[0:choice] = 1
        self.mask = mask.bool()

    @property
    def current_mask(self) -> torch.Tensor:
        """Return current mask."""
        return self.mask

    # methods for

    def fix_chosen(self, chosen=...):
        """Fix chosen."""
        if chosen is ...:
            chosen = self.current_choice
        assert self.is_fixed is False
        self.current_choice = chosen
        self.is_fixed = True

    def dump_chosen(self):
        """Dump chosen."""
        return self.current_choice

    # def __mul__(self, other):
    #     """multiplication."""
    #     if isinstance(other, int):
    #         return self.derive_expand_mutable(other)
    #     else:
    #         return None

    # def __floordiv__(self, other):
    #     """division."""
    #     if isinstance(other, int):
    #         return self.derive_divide_mutable(other)
    #     else:
    #         return None

    def __rmul__(self, other) -> DerivedMutable:
        return self * other

    def __mul__(self, other) -> DerivedMutable:
        if isinstance(other, int):
            return self.derive_expand_mutable(other)

        from ..mutable_value import OneShotMutableValue

        def expand_choice_fn(mutable1: 'SquentialMutableChannel',
                             mutable2: OneShotMutableValue) -> Callable:

            def fn():
                return mutable1.current_choice * mutable2.current_choice

            return fn

        def expand_mask_fn(mutable1: 'SquentialMutableChannel',
                           mutable2: OneShotMutableValue) -> Callable:

            def fn():
                mask = mutable1.current_mask
                max_expand_ratio = mutable2.max_choice
                current_expand_ratio = mutable2.current_choice
                expand_num_channels = mask.size(0) * max_expand_ratio

                expand_choice = mutable1.current_choice * current_expand_ratio
                expand_mask = torch.zeros(expand_num_channels).bool()
                expand_mask[:expand_choice] = True

                return expand_mask

            return fn

        if isinstance(other, OneShotMutableValue):
            return DerivedMutable(
                choice_fn=expand_choice_fn(self, other),
                mask_fn=expand_mask_fn(self, other))

        raise TypeError(f'Unsupported type {type(other)} for mul!')

    def __floordiv__(self, other) -> DerivedMutable:
        if isinstance(other, int):
            return self.derive_divide_mutable(other)
        if isinstance(other, tuple):
            assert len(other) == 2
            return self.derive_divide_mutable(*other)

        raise TypeError(f'Unsupported type {type(other)} for div!')
