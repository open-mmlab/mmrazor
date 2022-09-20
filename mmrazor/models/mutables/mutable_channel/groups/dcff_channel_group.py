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

    def __init__(self,
                 num_channels,
                 candidate_choices: List[Union[int, float]] = [32],
                 candidate_mode: str = 'number') -> None:
        super().__init__(num_channels, candidate_choices, candidate_mode)

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: FuseConv2d,
                nn.BatchNorm2d: DynamicBatchNorm2d,
                nn.Linear: DynamicLinear
            })
        self._register_mask_container(model, MutableChannelContainer)
        self._register_mask(self.mutable_channel)

    def _prepare_choices(self):
        for choice in self.candidate_choices:
            assert isinstance(choice, self.choice_type)
        self.candidate_choices = sorted(self.candidate_choices)

    def alter_candidates_after_init(self, candidates):
        self.candidate_choices = candidates
        self._prepare_choices()  # TODO refactor
        for channel in self.input_related:
            if isinstance(channel.module, FuseConv2d):
                channel.module.change_mutable_attrs_after_init(
                    'in_channels', candidates)
        for channel in self.output_related:
            if isinstance(channel.module, FuseConv2d):
                channel.module.change_mutable_attrs_after_init(
                    'out_channels', candidates)
