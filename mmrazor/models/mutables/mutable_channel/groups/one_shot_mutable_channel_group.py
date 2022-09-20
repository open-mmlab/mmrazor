# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from typing import Dict, List, Union

import torch.nn as nn

from mmrazor.registry import MODELS
from .sequential_mutable_channel_group import SequentialMutableChannelGroup


@MODELS.register_module()
class OneShotMutableChannelGroup(SequentialMutableChannelGroup):
    """OneShotMutableChannelGroup is for single path supernet such as AutoSlim.
    In single path supernet, each module only has one choice invoked at the
    same time. A path is obtained by sampling all the available choices. It is
    the base class for one shot mutable channel.

    Args:
        num_channels (_type_): The raw number of channels.
        candidate_choices (List[Union[int, float]], optional):
            A list of candidate width ratios. Each
            candidate indicates how many channels to be reserved.
            Defaults to [0.5, 1.0](candidate_mode='ratio').
        candidate_mode (str, optional): Mode of candidates.
            One of "ratio" or "number". Defaults to 'ratio'.
        divisor (int): Used to make choice divisible.
        min_value (int): the minimal value used when make divisible.
        min_ratio (float): the minimal ratio used when make divisible.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List[Union[int, float]] = [0.5, 1.0],
                 candidate_mode='ratio',
                 divisor=1,
                 min_value=1,
                 min_ratio=0.9) -> None:
        super().__init__(num_channels, candidate_mode, divisor, min_value,
                         min_ratio)
        candidate_choices = copy.copy(candidate_choices)
        if candidate_choices == []:
            candidate_choices.append(
                self.num_channels if self.is_num_mode else 1.0)
        self.candidate_choices = self._prepare_candidate_choices(
            candidate_choices, candidate_mode)

        self._choice = self.max_choice

    def prepare_for_pruning(self, model: nn.Module):
        """Prepare for pruning."""
        super().prepare_for_pruning(model)
        self.current_choice = self.max_choice

    # ~

    def config_template(self,
                        with_init_args=False,
                        with_channels=False) -> Dict:
        """Config template of the OneShotMutableChannelGroup."""
        config = super().config_template(with_init_args, with_channels)
        if with_init_args:
            init_cfg = config['init_args']
            init_cfg.pop('choice_mode')
            init_cfg.update({
                'candidate_choices': self.candidate_choices,
                'candidate_mode': self.choice_mode
            })
        return config

    # choice

    @property
    def current_choice(self) -> Union[int, float]:
        """Get current choice."""
        return super().current_choice

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """Set current choice."""
        assert choice in self.candidate_choices
        SequentialMutableChannelGroup.current_choice.fset(  # type: ignore
            self,  # type: ignore
            choice)  # type: ignore

    def sample_choice(self) -> Union[int, float]:
        """Sample a valid choice."""
        rand_idx = random.randint(0, len(self.candidate_choices) - 1)
        return self.candidate_choices[rand_idx]

    @property
    def min_choice(self) -> Union[int, float]:
        """Get Minimal choice."""
        return self.candidate_choices[0]

    @property
    def max_choice(self) -> Union[int, float]:
        """Get Maximal choice."""
        return self.candidate_choices[-1]

    # private methods

    def _prepare_candidate_choices(self, candidate_choices: List,
                                   candidate_mode) -> List:
        """Process candidate_choices."""
        choice_type = int if candidate_mode == 'number' else float
        for choice in candidate_choices:
            assert isinstance(choice, choice_type)
        if self.is_num_mode:
            candidate_choices_ = [
                self._make_divisible(choice) for choice in candidate_choices
            ]
        else:
            candidate_choices_ = [
                self._num2ratio(self._make_divisible(self._ratio2num(choice)))
                for choice in candidate_choices
            ]
        if candidate_choices_ != candidate_choices:
            self._make_divisible_info(candidate_choices, candidate_choices_)

        candidate_choices_ = sorted(candidate_choices_)
        return candidate_choices_
