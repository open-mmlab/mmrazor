# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import torch

from mmrazor.registry import MODELS
from ..derived_mutable import DerivedMutable
from .simple_mutable_channel import SimpleMutableChannel

# TODO discuss later


@MODELS.register_module()
class SquentialMutableChannel(SimpleMutableChannel):
    """SquentialMutableChannel defines a BaseMutableChannel which switch off
    channel mask from right to left sequentially, like '11111000'.

    A choice of SquentialMutableChannel is an integer, which indicates how many
    channel are activated from left to right.

    Args:
        num_channels (int): number of channels.
    """

    def __init__(self, num_channels: int, choice_mode='number', **kwargs):

        super().__init__(num_channels, **kwargs)
        assert choice_mode in ['ratio', 'number']
        self.choice_mode = choice_mode
        self.mask = torch.ones([self.num_channels]).bool()

    @property
    def is_num_mode(self):
        """Get if the choice is number mode."""
        return self.choice_mode == 'number'

    @property
    def current_choice(self) -> Union[int, float]:
        """Get current choice."""
        int_choice = (self.mask == 1).sum().item()
        if self.is_num_mode:
            return int_choice
        else:
            return self._num2ratio(int_choice)

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """Set choice."""
        if isinstance(choice, float):
            int_choice = self._ratio2num(choice)
        else:
            int_choice = choice
        mask = torch.zeros([self.num_channels], device=self.mask.device)
        mask[0:int_choice] = 1
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

    def __rmul__(self, other) -> DerivedMutable:
        return self * other

    def __mul__(self, other) -> DerivedMutable:
        if isinstance(other, int) or isinstance(other, float):
            return self.derive_expand_mutable(other)

        from ..mutable_value import OneShotMutableValue

        def expand_choice_fn(mutable1: 'SquentialMutableChannel',
                             mutable2: OneShotMutableValue) -> Callable:

            def fn():
                return int(mutable1.current_choice * mutable2.current_choice)

            return fn

        def expand_mask_fn(mutable1: 'SquentialMutableChannel',
                           mutable2: OneShotMutableValue) -> Callable:

            def fn():
                mask = mutable1.current_mask
                max_expand_ratio = mutable2.max_choice
                current_expand_ratio = mutable2.current_choice
                expand_num_channels = int(mask.size(0) * max_expand_ratio)

                expand_choice = int(mutable1.current_choice *
                                    current_expand_ratio)
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
        elif isinstance(other, float):
            return self.derive_divide_mutable(int(other))
        if isinstance(other, tuple):
            assert len(other) == 2
            return self.derive_divide_mutable(*other)

        from ..mutable_value import OneShotMutableValue
        if isinstance(other, OneShotMutableValue):
            ratio = other.current_choice
            return self.derive_divide_mutable(ratio)

        raise TypeError(f'Unsupported type {type(other)} for div!')

    def _num2ratio(self, choice: Union[int, float]) -> float:
        """Convert the a number choice to a ratio choice."""
        if isinstance(choice, float):
            return choice
        else:
            return choice / self.num_channels

    def _ratio2num(self, choice: Union[int, float]) -> int:
        """Convert the a ratio choice to a number choice."""
        if isinstance(choice, int):
            return choice
        else:
            return max(1, int(self.num_channels * choice))
