# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch

from mmrazor.registry import MODELS
from ..derived_mutable import DerivedMutable
from .base_mutable_channel import BaseMutableChannel


@MODELS.register_module()
class SimpleMutableChannel(BaseMutableChannel):

    def __init__(self, num_channels, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = num_channels
        self.mask = torch.ones(num_channels).bool()

    # choice

    @property
    def current_choice(self):
        return self.mask

    @current_choice.setter
    def current_choice(self, choice: torch.Tensor):
        self.mask = choice

    @property
    def current_mask(self):
        return self.current_choice

    # basic extension

    def expand_mutable_mask(self, expand_ratio):
        derive_fun = partial(
            _expand_mask, mutable_mask=self, expand_ratio=expand_ratio)
        return DerivedMutable(derive_fun, derive_fun, [self])

    # others

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '('
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'activated_channels: {self.activated_channels}'
        repr_str += ')'
        return repr_str


def _expand_mask(mutable_mask, expand_ratio):
    mask = mutable_mask.current_mask
    mask = torch.unsqueeze(mask, -1).expand(list(mask.shape) +
                                            [expand_ratio]).flatten(-2)
    return mask
