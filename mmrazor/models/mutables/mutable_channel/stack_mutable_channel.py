# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS
from .base_mutable_channel import BaseMutableChannel


@MODELS.register_module()
class StackMutableChannel(BaseMutableChannel):

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.mask = torch.ones([self.num_channels]).bool()

    @property
    def current_choice(self):
        return (self.mask == 1).sum().item()

    @current_choice.setter
    def current_choice(self, choice: int):
        mask = torch.zeros([self.num_channels])
        mask[0:choice] = 1
        self.mask = mask.bool()

    @property
    def current_mask(self) -> torch.Tensor:
        return self.mask

    #

    def fix_chosen(self, chosen):
        assert self.is_fixed is False
        self.current_choice = chosen
        self.is_fixed = True

    def dump_chosen(self):
        return self.current_choice

    def __mul__(self, other):
        if isinstance(other, int):
            return self.derive_expand_mutable(other)
        else:
            return None

    def __floordiv__(self, other):
        if isinstance(other, int):
            return self.derive_divide_mutable(other)
        else:
            return None


def int2mask(total_channel, activated_num):
    mask = torch.zeros([total_channel])
    mask[0:activated_num] = 1
    return mask.bool()
