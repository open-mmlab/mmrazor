# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict, Union

import torch
import torch.nn as nn
from mmengine import MMLogger

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicBatchNorm2d, DynamicConv2d, DynamicLinear)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from ..simple_mutable_channel import SimpleMutableChannel
from .mutable_channel_group import MutableChannelGroup


# TODO change the name of SequentialMutableChannelGroup
@MODELS.register_module()
class SequentialMutableChannelGroup(MutableChannelGroup):
    """SequentialMutableChannelGroup accepts a intger(number) or float(ratio)
    as the choice, which indicates how many of the channels are remained from
    left to right, like 11110000.

    Args:
        num_channels (int): number of channels.
        choice_mode (str): mode of choice, which is one of 'number' or 'ratio'.
    """

    def __init__(self,
                 num_channels: int,
                 choice_mode='number',
                 divisor=1) -> None:
        super().__init__(num_channels)
        self.mutable_channel: SimpleMutableChannel = SimpleMutableChannel(
            self.num_channels)
        assert choice_mode in ['ratio', 'number']
        self.choice_mode = choice_mode
        self.divisor = divisor

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

    #  ~

    @property
    def is_num_mode(self):
        return self.choice_mode == 'number'

    def fix_chosen(self, choice=None):
        """fix chosen."""
        super().fix_chosen(choice)
        self.mutable_channel.fix_chosen()

    def config_template(self,
                        with_init_args=False,
                        with_channels=False) -> Dict:
        """Template of config."""
        config = super().config_template(with_init_args, with_channels)
        if with_init_args:
            init_args: Dict = config['init_args']
            init_args.update(
                dict(choice_mode=self.choice_mode, divisor=self.divisor))
        return config

    # choice

    @property
    def current_choice(self) -> Union[int, float]:
        """return current choice."""
        if self.is_num_mode:
            return self.mutable_channel.activated_channels
        else:
            return self._num2ratio(self.mutable_channel.activated_channels)

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """set choice."""
        choice_num = self._ratio2num(choice)
        choice_num_ = self._make_divisible(choice_num)

        mask = self._generate_mask(choice_num_)
        self.mutable_channel.current_choice = mask
        if choice_num != choice_num_:
            logger = MMLogger.get_current_instance()
            logger.info(
                f'The choice={choice}, which is set to {self.name}, '
                f'is changed to {self.current_choice} for a divisible choice.')

    def sample_choice(self) -> Union[int, float]:
        """Sample a choice in (0,1]"""
        num_choice = random.randint(1, self.num_channels)
        num_choice = self._make_divisible(num_choice)
        if self.is_num_mode:
            return num_choice
        else:
            return self._num2ratio(num_choice)

    # private methods

    def _make_divisible(self, choice_int: int):
        """forked from slim:

        https://github.com/tensorflow/models/blob/\
        0344c5503ee55e24f0de7f37336a6e08f10976fd/\
        research/slim/nets/mobilenet/mobilenet.py#L62-L69
        """
        new_choice = max(
            self.divisor,
            int(choice_int + self.divisor / 2) // self.divisor * self.divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_choice < 0.9 * choice_int:
            new_choice += self.divisor
        return new_choice

    def _num2ratio(self, choice: Union[int, float]) -> float:
        """Convert the a number choice to a ratio choice."""
        if isinstance(choice, float):
            return choice
        else:
            return choice / self.num_channels

    def _ratio2num(self, choice: Union[int, float]) -> int:
        """Convert the a ratio choice to a number choice."""
        if isinstance(choice, int):
            return choice
        else:
            return max(1, int(self.num_channels * choice))

    def _generate_mask(self, choice: int) -> torch.Tensor:
        """torch.Tesnor: generate mask for pruning"""
        mask = torch.zeros([self.num_channels])
        mask[0:choice] = 1
        return mask
