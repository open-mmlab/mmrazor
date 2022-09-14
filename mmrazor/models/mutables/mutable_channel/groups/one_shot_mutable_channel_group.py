# Copyright (c) OpenMMLab. All rights reserved.
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
    """

    def __init__(
        self,
        num_channels,
        candidate_choices: List[Union[int, float]] = [0.5, 1.0],
        candidate_mode: str = 'ratio',
    ) -> None:

        super().__init__(num_channels)

        assert candidate_mode in ['ratio', 'number']
        self.candidate_mode = candidate_mode
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
            init_cfg.update({
                'candidate_choices': self.candidate_choices,
                'candidate_mode': self.candidate_mode
            })
        return config

    # choice

    @property
    def current_choice(self) -> Union[int, float]:
        """Get current choice."""
        return self._choice

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """Set current choice."""
        assert choice in self.candidate_choices
        self._choice = choice
        choice_int = self._get_int_choice(choice)

        SequentialMutableChannelGroup.current_choice.fset(  # type: ignore
            self,  # type: ignore
            choice_int)  # type: ignore

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
        candidate_choices = sorted(candidate_choices)
        return candidate_choices
