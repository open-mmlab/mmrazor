# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch

from mmrazor.registry import MODELS
from ..derived_mutable import DerivedMutable
from .base_mutable_channel import BaseMutableChannel


@MODELS.register_module()
class SimpleMutableChannel(BaseMutableChannel):
    """SimpleMutableChannel is a simple BaseMutableChannel, it directly take a
    mask as a choice."""

    def __init__(self, num_channels, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = num_channels
        self.mask = torch.ones(num_channels).bool()

    # choice

    @property
    def current_choice(self) -> torch.Tensor:
        """Get current choice."""
        return self.mask.bool()

    @current_choice.setter
    def current_choice(self, choice: torch.Tensor):
        """Set current choice."""
        self.mask = choice.to(self.mask.device).bool()

    @property
    def current_mask(self) -> torch.Tensor:
        """Get current mask."""
        return self.current_choice.bool()

    # basic extension

    def expand_mutable_channel(self, expand_ratio) -> DerivedMutable:
        """Get a derived SimpleMutableChannel with expanded mask."""

        def _expand_mask(mutable_channel, expand_ratio):
            mask = mutable_channel.current_mask
            mask = torch.unsqueeze(
                mask, -1).expand(list(mask.shape) + [expand_ratio]).flatten(-2)
            return mask

        derive_fun = partial(
            _expand_mask, mutable_channel=self, expand_ratio=expand_ratio)
        return DerivedMutable(derive_fun, derive_fun, [self])

    # others

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '('
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'activated_channels: {self.activated_channels}'
        repr_str += ')'
        return repr_str
