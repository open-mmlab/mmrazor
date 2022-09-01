# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn

from mmrazor.models.architectures.dynamic_op.bricks import (DynamicBatchNorm2d,
                                                            DynamicLinear,
                                                            FuseConv2d)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .one_shot_channel_group import OneShotChannelGroup


@MODELS.register_module()
class DCFFChannelGroup(OneShotChannelGroup):

    def __init__(self,
                 num_channels,
                 candidate_choices: List[Union[int, float]] = [0.5, 1.0],
                 candidate_mode: str = 'ratio') -> None:
        super().__init__(num_channels, candidate_choices, candidate_mode)

    @classmethod
    def prepare_model(cls, model: nn.Module):
        cls._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: FuseConv2d,
                nn.BatchNorm2d: DynamicBatchNorm2d,
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
            if isinstance(channel.module, DynamicBatchNorm2d) and \
                    len(channel.module.candidate_bn) == 0:
                channel.module.init_candidates(candidates)
