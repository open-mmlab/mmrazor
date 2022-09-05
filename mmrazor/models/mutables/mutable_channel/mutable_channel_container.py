# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch

from mmrazor.registry import MODELS
from mmrazor.utils import IndexDict
from .base_mutable_channel import BaseMutableChannel
from .simple_mutable_channel import SimpleMutableChannel


@MODELS.register_module()
class MutableChannelContainer(BaseMutableChannel):
    """MutableChannelContainer inherits from BaseMutableChannel. However,
    it's not a single BaseMutableChannel, but a container for
    BaseMutableChannel. The mask of MutableChannelContainer consists of
    all masks of stored MutableChannels.

    -----------------------------------------------------------
    |                   MutableChannelContainer               |
    -----------------------------------------------------------
    |MutableChannel1|     MutableChannel2     |MutableChannel3|
    -----------------------------------------------------------

    Important interfaces:
        register_mutable: register/store BaseMutableChannel in the
            MutableChannelContainer
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.mutable_channels: IndexDict[BaseMutableChannel] = IndexDict()

    # choice

    @property
    def current_choice(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return torch.ones([self.num_channels]).bool()
        else:
            self._full_empty_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            masks = [mutable.current_mask for mutable in mutable_channels]
            mask = torch.cat(masks)
            return mask.bool()

    @current_choice.setter
    def current_choice(self, choice):
        """Set current choices.

        However, MutableChannelContainer doesn't support directly set mask. You
        can change the mask of MutableChannelContainer by changing its stored
        BaseMutableChannel.
        """
        raise NotImplementedError()

    @property
    def current_mask(self) -> torch.Tensor:
        """Return current mask."""
        return self.current_choice.bool()

    # basic extension

    def register_mutable(self, mutable_channel: BaseMutableChannel, start: int,
                         end: int):
        """Register/Store BaseMutableChannel in the MutableChannelContainer  in
        the range [start,end)"""

        self.mutable_channels[(start, end)] = mutable_channel

    # private methods

    def _assert_mutables_valid(self):
        """Assert the current stored BaseMutableChannels are valid to generate
        mask."""
        assert len(self.mutable_channels) > 0
        last_end = 0
        for start, end in self.mutable_channels:
            assert start == last_end
            last_end = end
        assert last_end == self.num_channels

    def _full_empty_range(self):
        """Add SimpleMutableChannels in the range without any stored
        BaseMutableChannel."""
        last_end = 0
        for start, end in copy.copy(self.mutable_channels):
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
