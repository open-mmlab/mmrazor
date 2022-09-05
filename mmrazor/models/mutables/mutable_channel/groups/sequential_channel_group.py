# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicBatchNorm2d, DynamicConv2d, DynamicLinear)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from ..simple_mutable_channel import SimpleMutableChannel
from .mutable_channel_group import MutableChannelGroup


@MODELS.register_module()
class SequentialChannelGroup(MutableChannelGroup):
    """SimpleChannelGroup defines a simple pruning algorithhm.

    The type of choice of SimpleChannelGroup is float. It indicates what ratio
    of channels are remained from left to right.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.mutable_channel: SimpleMutableChannel = SimpleMutableChannel(
            self.num_channels)

    # prepare model

    def prepare_for_pruning(self, model: nn.Module):
        """Prepare for pruning, including register mutable channels."""
        # register MutableMask
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: DynamicConv2d,
                nn.BatchNorm2d: DynamicBatchNorm2d,
                nn.Linear: DynamicLinear
            })
        self._register_mask_container(model, MutableChannelContainer)
        self._register_mask(self.mutable_channel)

    # choice

    @property
    def current_choice(self) -> float:
        """return current choice."""
        return self.mutable_channel.activated_channels / self.num_channels

    @current_choice.setter
    def current_choice(self, choice: float):
        """set choice."""
        int_choice = self._get_int_choice(choice)
        mask = self._generate_mask(int_choice)
        self.mutable_channel.current_choice = mask

    def sample_choice(self) -> float:
        """Sample a choice in (0,1]"""
        return max(1, int(
            random.random() * self.num_channels)) / self.num_channels

    # private methods

    def _generate_mask(self, choice: int) -> torch.Tensor:
        """torch.Tesnor: generate mask for pruning"""
        mask = torch.zeros([self.num_channels])
        mask[0:choice] = 1
        return mask

    # interface
    def fix_chosen(self, choice=None):
        """fix chosen."""
        super().fix_chosen(choice)
        self.mutable_channel.fix_chosen()
