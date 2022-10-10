# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

from .sequential_mutable_channel import SquentialMutableChannel


class OneShotMutableChannel(SquentialMutableChannel):

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List = [],
                 choice_mode='number',
                 **kwargs):
        super().__init__(num_channels, choice_mode, **kwargs)
        self.candidate_choices = candidate_choices
        if candidate_choices == []:
            candidate_choices.append(num_channels if self.is_num_mode else 1.0)

    @property
    def current_choice(self) -> Union[int, float]:
        """Get current choice."""
        return super().current_choice

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """Set current choice."""
        assert choice in self.candidate_choices
        SquentialMutableChannel.current_choice.fset(  # type: ignore
            self,  # type: ignore
            choice)  # type: ignore
