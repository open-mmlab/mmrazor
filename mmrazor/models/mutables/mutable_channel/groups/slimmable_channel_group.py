# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicConv2d, DynamicLinear, SwitchableBatchNorm2d)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .one_shot_channel_group import OneShotChannelGroup


@MODELS.register_module()
class SlimmableChannelGroup(OneShotChannelGroup):

    def __init__(self, num_channels, candidate_choices=...) -> None:
        if candidate_choices == ...:
            candidate_choices = [num_channels]
        super().__init__(
            num_channels, candidate_choices, candidate_mode='number')
        self.init_args = dict(candidate_choices=candidate_choices)

    def config_template(self, with_info=False):
        config = super().config_template(with_info)
        config.pop('candidate_mode')
        return config

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: DynamicConv2d,
                nn.BatchNorm2d: SwitchableBatchNorm2d,
                nn.Linear: DynamicLinear
            })
        self._register_mask_container(model, MutableChannelContainer)
        self._register_mask(self.mutable_channel)

    def alter_candidates_after_init(self, candidates):
        self.candidate_choices = candidates
        self._prepare_choices()  # TODO refactor
        for channel in self.output_related:
            if isinstance(channel.module, SwitchableBatchNorm2d) and \
                    len(channel.module.candidate_bn) == 0:
                channel.module.init_candidates(candidates)
        self.current_choice = self.max_choice
