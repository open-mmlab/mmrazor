# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_op.bricks import (DynamicBatchNorm2d,
                                                            DynamicConv2d,
                                                            DynamicLinear)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from ..simple_mutable_channel import SimpleMutableChannel
from .mutable_channel_group import MutableChannelGroup


@MODELS.register_module()
class SimpleChannelGroup(MutableChannelGroup):

    def __init__(self, num_channels) -> None:
        super().__init__(num_channels)
        self.mutable_mask: SimpleMutableChannel

    # choice

    @property
    def current_choice(self) -> float:
        return self.mutable_mask.activated_channels / self.num_channels

    @current_choice.setter
    def current_choice(self, choice: float):
        """Current choice setter will be executed in mutator."""
        int_choice = self._get_int_choice(choice)
        mask = self._generate_mask(int_choice)
        self.mutable_mask.current_choice = mask

    def sample_choice(self):
        return max(1, int(
            random.random() * self.num_channels)) / self.num_channels

    # prepare model

    @classmethod
    def prepare_model(
        cls,
        model: nn.Module,
    ):
        cls._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: DynamicConv2d,
                nn.BatchNorm2d: DynamicBatchNorm2d,
                nn.Linear: DynamicLinear
            })

        # register MutableMaskContainer
        cls._register_mask_container(model, MutableChannelContainer)

    def prepare_for_pruning(self):
        self.mutable_mask = SimpleMutableChannel(self.num_channels)

        # register MutableMask
        self._register_mask(self.mutable_mask)

    # private methods

    def _generate_mask(self, choice: int) -> torch.Tensor:
        """torch.Tesnor: generate mask for pruning"""
        mask = torch.zeros([self.num_channels])
        mask[0:choice] = 1
        return mask
