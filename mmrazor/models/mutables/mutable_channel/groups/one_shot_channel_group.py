# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Union

import torch

from .....registry import MODELS
from .simple_channel_group import SimpleChannelGroup


@MODELS.register_module()
class OneShotChannelGroup(SimpleChannelGroup):

    def __init__(
        self,
        num_channels,
        candidate_choices: List[Union[int, float]] = [0.5, 1.0],
        candidate_mode: str = 'ratio',
    ) -> None:
        super().__init__(num_channels)

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
        self.mutable_mask.current_choice = mask

    def sample_choice(self):
        rand_idx = random.randint(0, len(self.candidate_choices) - 1)
        return self.candidate_choices[rand_idx]

    @property
    def min_choice(self) -> Union[int, float]:
        return self.candidate_choices[0]

    @property
    def max_choice(self) -> Union[int, float]:
        return self.candidate_choices[-1]

    # prepare
    def prepare_for_pruning(self):
        super().prepare_for_pruning()
        self._choice = self.max_choice
        self.current_choice = self._choice

    # private methods
    def _prepare_choices(self):
        for choice in self.candidate_choices:
            assert isinstance(choice, self.choice_type)
        self.candidate_choices = sorted(self.candidate_choices)

    def _generate_mask(self, num_activated_channels):
        mask = torch.zeros([self.num_channels])
        mask[0:num_activated_channels] = 1
        return mask.bool()
