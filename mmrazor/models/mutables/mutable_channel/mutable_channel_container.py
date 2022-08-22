# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch

from ....registry import MODELS
from ....utils.index_dict import IndexDict
from .mutable_channel import MutableChannel
from .simple_mutable_channel import SimpleMutableChannel


@MODELS.register_module()
class MutableChannelContainer(MutableChannel):

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.name = ''
        self.num_channels = num_channels
        self.mutable_masks: IndexDict[SimpleMutableChannel] = IndexDict()

    # choice

    @property
    def current_choice(self):
        if len(self.mutable_masks) == 0:
            return torch.ones([self.num_channels]).bool()
        else:
            self._full_with_empty_mask()
            self._assert_mask_valid()
            mutable_masks = list(self.mutable_masks.values())
            masks = [mutable.current_mask for mutable in mutable_masks]
            mask = torch.cat(masks)
            return mask

    @current_choice.setter
    def current_choice(self, choice):
        raise NotImplementedError()

    @property
    def current_mask(self):
        return self.current_choice

    # basic extension

    @property
    def activated_channels(self):
        return (self.current_mask == 1).sum().item()

    def register_mutable(self, mutable_mask: SimpleMutableChannel, start: int,
                         end: int):
        self.mutable_masks[(start, end)] = mutable_mask

    # private methods

    def _assert_mask_valid(self):
        assert len(self.mutable_masks) > 0
        last_end = 0
        for start, end in self.mutable_masks:
            assert start == last_end
            last_end = end
        assert last_end == self.num_channels

    def _full_with_empty_mask(self):
        last_end = 0
        for start, end in copy.copy(self.mutable_masks):
            if last_end < start:
                self.register_mutable(
                    SimpleMutableChannel(last_end - start), last_end, start)
            last_end = end
        if last_end < self.num_channels:
            self.register_mutable(
                SimpleMutableChannel(self.num_channels - last_end), last_end,
                self.num_channels)

    # others

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'activated_channels: {self.activated_channels}'
        return repr_str

    # implement abstract methods

    def num_choices(self) -> int:
        return self.num_channels

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen) -> None:
        pass

    def convert_choice_to_mask(self, choice) -> torch.Tensor:
        return self.current_choice
