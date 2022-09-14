# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Union

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicBatchNorm2d, DynamicConv2d, DynamicLinear)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from ..simple_mutable_channel import SimpleMutableChannel
from .mutable_channel_group import MutableChannelGroup


# TODO change the name of SequentialMutableChannelGroup
@MODELS.register_module()
class SequentialMutableChannelGroup(MutableChannelGroup):
    """SequentialMutableChannelGroup accepts a intger as the choice, which
    indicates the number of the channels are remained from left to right, like
    11110000.

    Args:
        num_channels (int): number of channels.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.mutable_channel: SimpleMutableChannel = SimpleMutableChannel(
            self.num_channels)

    def prepare_for_pruning(self, model: nn.Module):
        """Prepare for pruning, including register mutable channels."""
        # register MutableMask
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: DynamicConv2d,
                nn.BatchNorm2d: DynamicBatchNorm2d,
                nn.Linear: DynamicLinear
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    # choice

    @property
    def current_choice(self) -> Union[int, float]:
        """return current choice."""
        return self.mutable_channel.activated_channels

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """set choice."""
        assert 0 < choice <= self.num_channels
        mask = self._generate_mask(choice)
        self.mutable_channel.current_choice = mask

    def sample_choice(self) -> int:
        """Sample a choice in (0,1]"""
        return random.randint(1, self.num_channels)

    # private methods

    def _generate_mask(self, choice: int) -> torch.Tensor:
        """torch.Tesnor: generate mask for pruning"""
        mask = torch.zeros([self.num_channels])
        mask[0:choice] = 1
        return mask

    def fix_chosen(self, choice=None):
        """fix chosen."""
        super().fix_chosen(choice)
        self.mutable_channel.fix_chosen()
