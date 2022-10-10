# Copyright (c) OpenMMLab. All rights reserved.
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

    def min_choices(self) -> Dict:
        """Return the minimal pruning subnet(structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2unit[key].min_choice
        return template

    def max_choices(self) -> Dict:
        """Return the maximal pruning subnet(structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2unit[key].max_choice
        return template
