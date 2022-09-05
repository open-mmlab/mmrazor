# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS
from .base_mutable_channel import BaseMutableChannel


@MODELS.register_module()
class SquentialMutableChannel(BaseMutableChannel):
    """SquentialMutableChannel defines a BaseMutableChannel which switch off
    channel mask from right to left sequentially, like '11111000'.

    A choice of SquentialMutableChannel is an integer, which indicates how many
    channel are activated from left to right.
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

    def __mul__(self, other):
        """multiplication."""
        if isinstance(other, int):
            return self.derive_expand_mutable(other)
        else:
            return None

    def __floordiv__(self, other):
        """division."""
        if isinstance(other, int):
            return self.derive_divide_mutable(other)
        else:
            return None
