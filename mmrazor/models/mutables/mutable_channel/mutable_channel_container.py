# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch

from mmrazor.registry import MODELS
from mmrazor.utils import IndexDict
from ...architectures.dynamic_ops.mixins import DynamicChannelMixin
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
        self.mutable_channels = IndexDict()

    # choice

    @property
    def current_choice(self) -> torch.Tensor:
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return torch.ones([self.num_channels]).bool()
        else:
            self._fill_unregistered_range()
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
        """Register/Store BaseMutableChannel in the MutableChannelContainer in
        the range [start,end)"""

        self.mutable_channels[(start, end)] = mutable_channel

    @classmethod
    def register_mutable_channel_to_module(cls,
                                           module: DynamicChannelMixin,
                                           mutable: BaseMutableChannel,
                                           is_to_output_channel=True,
                                           start=0,
                                           end=-1):
        """Register a BaseMutableChannel to a module with
        MutableChannelContainers."""
        if end == -1:
            end = mutable.current_choice + start
        if is_to_output_channel:
            container: MutableChannelContainer = module.get_mutable_attr(
                'out_channels')
        else:
            container = module.get_mutable_attr('in_channels')
        assert isinstance(container, MutableChannelContainer)
        container.register_mutable(mutable, start, end)

    # private methods

    def _assert_mutables_valid(self):
        """Assert the current stored BaseMutableChannels are valid to generate
        mask."""
        assert len(self.mutable_channels) > 0
        last_end = 0
        for start, end in self.mutable_channels:
            assert start == last_end
            last_end = end
        assert last_end == self.num_channels, (
            f'channel mismatch: {last_end} vs {self.num_channels}')

    def _fill_unregistered_range(self):
        """Fill with SimpleMutableChannels in the range without any stored
        BaseMutableChannel.

        For example, if a MutableChannelContainer has 10 channels, and only the
        [0,5) is registered with BaseMutableChannels, this method will
        automatically register BaseMutableChannels in the range [5,10).
        """
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
