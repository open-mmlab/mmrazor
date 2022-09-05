# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .sequential_channel_group import SequentialChannelGroup


@MODELS.register_module()
class OneShotChannelGroup(SequentialChannelGroup):

    def __init__(
        self,
        num_channels,
        candidate_choices: List[Union[int, float]] = [0.5, 1.0],
        candidate_mode: str = 'ratio',
    ) -> None:
        super().__init__(num_channels)
        self.init_args = dict(
            candidate_choices=candidate_choices, candidate_mode=candidate_mode)

        self.candidate_choices = candidate_choices
        self.candidate_mode = candidate_mode
        assert candidate_mode in ['ratio', 'number']
        self.choice_type = int if candidate_mode == 'number' else float

        self._prepare_choices()

    # choice

    @property
    def current_choice(self) -> Union[int, float]:
        return self._choice

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        assert choice in self.candidate_choices
        self._choice = choice
        activated_channels = choice if self.choice_type is int else int(
            self.num_channels * choice)
        mask = self._generate_mask(activated_channels)
        self.mutable_channel.current_choice = mask

    def sample_choice(self):
        rand_idx = random.randint(0, len(self.candidate_choices) - 1)
        return self.candidate_choices[rand_idx]

    @property
    def min_choice(self) -> Union[int, float]:
        return self.candidate_choices[0]

    @property
    def max_choice(self) -> Union[int, float]:
        return self.candidate_choices[-1]

    def config_template(self, with_info=False):
        config = super().config_template(with_info=with_info)
        config.update({
            'candidates': self.candidate_choices,
            'candidate_mode': self.candidate_mode
        })
        return config

    # prepare
    def prepare_for_pruning(self, model: nn.Module):
        super().prepare_for_pruning(model)
        self._choice = self.max_choice
        self.current_choice = self._choice

    # private methods

    def _prepare_choices(self):
        for choice in self.candidate_choices:
            assert isinstance(choice, self.choice_type)
        self.candidate_choices = sorted(self.candidate_choices)

    def _generate_mask(self, num_activated_channels):
        mask = torch.zeros([self.num_channels], dtype=torch.bool)
        mask[0:num_activated_channels] = True
        return mask.bool()
