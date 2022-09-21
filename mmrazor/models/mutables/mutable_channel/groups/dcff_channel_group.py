# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicBatchNorm2d, DynamicLinear, FuseConv2d)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .one_shot_mutable_channel_group import OneShotMutableChannelGroup


@MODELS.register_module()
class DCFFChannelGroup(OneShotMutableChannelGroup):
    """``DCFFChannelGroup`` is for supernet DCFF and
    based on OneShotMutableChannelGroup.
    In DCFF supernet, each module only has one choice.
    The channel choice is fixed before training.

    Args:
        num_channels (int): The raw number of channels.
        candidate_choices (List[Union[int, float]], optional):
            A list of candidate width numbers or ratios. Each
            candidate indicates how many channels to be reserved.
            Defaults to [32](candidate_mode='number').
        candidate_mode (str, optional): Mode of candidates.
            One of "ratio" or "number". Defaults to 'number'.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List[Union[int, float]] = [32],
                 candidate_mode: str = 'number') -> None:
        super().__init__(num_channels, candidate_choices, candidate_mode)

    def prepare_for_pruning(self, model: nn.Module):
        """In ``DCFFChannelGroup`` nn.Conv2d is replaced with FuseConv2d.
        """
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: FuseConv2d,
                nn.BatchNorm2d: DynamicBatchNorm2d,
                nn.Linear: DynamicLinear
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def alter_candidates_after_init(self, candidates):
        self.candidate_choices = candidates
        for channel in self.input_related:
            if isinstance(channel.module, FuseConv2d):
                channel.module.change_mutable_attrs_after_init(
                    'in_channels', candidates)
        for channel in self.output_related:
            if isinstance(channel.module, FuseConv2d):
                channel.module.change_mutable_attrs_after_init(
                    'out_channels', candidates)
