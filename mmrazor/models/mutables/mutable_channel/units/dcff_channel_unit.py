# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .sequential_mutable_channel_unit import SequentialMutableChannelUnit


@MODELS.register_module()
class DCFFChannelUnit(SequentialMutableChannelUnit):
    """``DCFFChannelUnit`` is for supernet DCFF and based on
    OneShotMutableChannelUnit. In DCFF supernet, each module only has one
    choice. The channel choice is fixed before training.

    Args:
        num_channels (int): The raw number of channels.
        candidate_choices (List[Union[int, float]], optional):
            A list of candidate width numbers or ratios. Each
            candidate indicates how many channels to be reserved.
            Defaults to [1.0](choice_mode='number').
        choice_mode (str, optional): Mode of candidates.
            One of "ratio" or "number". Defaults to 'ratio'.
        divisor (int): Used to make choice divisible.
        min_value (int): the minimal value used when make divisible.
        min_ratio (float): the minimal ratio used when make divisible.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List[Union[int, float]] = [1.0],
                 choice_mode: str = 'ratio',
                 divisor: int = 1,
                 min_value: int = 1,
                 min_ratio: float = 0.9) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio)

    def prepare_for_pruning(self, model: nn.Module):
        """In ``DCFFChannelGroup`` nn.Conv2d is replaced with FuseConv2d."""
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: dynamic_ops.FuseConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: dynamic_ops.DynamicLinear
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)
