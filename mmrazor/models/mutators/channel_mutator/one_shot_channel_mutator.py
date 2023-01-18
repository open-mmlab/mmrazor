# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Type, Union

from mmrazor.models.mutables import OneShotMutableChannelUnit
from mmrazor.registry import MODELS
from .channel_mutator import ChannelMutator, ChannelUnitType


@MODELS.register_module()
class OneShotChannelMutator(ChannelMutator[OneShotMutableChannelUnit]):
    """OneShotChannelMutator based on ChannelMutator. It use
    OneShotMutableChannelUnit by default.

    Args:
        channel_unit_cfg (Union[dict, Type[ChannelUnitType]], optional):
            Config of MutableChannelUnits. Defaults to
            dict( type='OneShotMutableChannelUnit',
            default_args=dict( num_blocks=8, min_blocks=2 ) ).
    """

    def __init__(self,
                 channel_unit_cfg: Union[dict, Type[ChannelUnitType]] = dict(
                     type='OneShotMutableChannelUnit',
                     default_args=dict(num_blocks=8, min_blocks=2)),
                 **kwargs) -> None:

        super().__init__(channel_unit_cfg, **kwargs)

    @property
    def max_choices(self) -> Dict:
        """Get max choice for each unit in choice_template."""
        max_choices = copy.deepcopy(self.choice_template)
        for key in self.choice_template:
            max_choices[key] = self._name2unit[key].max_choice
        return max_choices

    @property
    def min_choices(self) -> Dict:
        """Get min choice for each unit in choice_template."""
        min_choices = copy.deepcopy(self.choice_template)
        for key in self.choice_template:
            min_choices[key] = self._name2unit[key].min_choice
        return min_choices

    def sample_choices(self, kind: str = 'random') -> Dict:
        """Sample choice for each unit in choice_template."""
        choices = copy.deepcopy(self.choice_template)
        for key in self.choice_template:
            if kind == 'max':
                choices[key] = self._name2unit[key].max_choice
            elif kind == 'min':
                choices[key] = self._name2unit[key].min_choice
            elif kind == 'random':
                choices[key] = self._name2unit[key].sample_choice()
            else:
                raise NotImplementedError()
        return choices

    def set_max_choices(self):
        """Set max choice for each unit in choice_template."""
        for name, choice in self.max_choices.items():
            unit = self._name2unit[name]
            unit.current_choice = choice

    def set_min_choices(self):
        """Set min choice for each unit in choice_template."""
        for name, choice in self.min_choices.items():
            unit = self._name2unit[name]
            unit.current_choice = choice
