# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmrazor.models.architectures.dynamic_op.bricks import (
    DynamicConv2d, DynamicLinear, SwitchableBatchNorm2d)
from .....registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .one_shot_channel_group import OneShotChannelGroup


@MODELS.register_module()
class SlimmableChannelGroup(OneShotChannelGroup):

    def __init__(self, num_channels, candidate_choices=[16, 32]) -> None:
        super().__init__(
            num_channels, candidate_choices, candidate_mode='number')

    @classmethod
    def prepare_model(cls, model: nn.Module):
        cls._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: DynamicConv2d,
                nn.BatchNorm2d: SwitchableBatchNorm2d,
                nn.Linear: DynamicLinear
            })

        # register MutableMaskContainer
        cls._register_mask_container(model, MutableChannelContainer)

    def prepare_for_pruning(self):
        super().prepare_for_pruning()

    def alter_candidates_after_init(self, candidates):
        self.candidate_choices = candidates
        self._prepare_choices()  # TODO refactor
        for channel in self.output_related:
            if isinstance(channel.module, SwitchableBatchNorm2d) and \
                    len(channel.module.candidate_bn) == 0:
                channel.module.init_candidates(candidates)
