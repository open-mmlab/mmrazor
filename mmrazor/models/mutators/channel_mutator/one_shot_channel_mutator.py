# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Type, Union

from mmrazor.models.mutables import OneShotMutableChannelGroup
from mmrazor.registry import MODELS
from .channel_mutator import ChannelGroupType, ChannelMutator


@MODELS.register_module()
class OneShotChannelMutator(ChannelMutator[OneShotMutableChannelGroup]):
    """OneShotChannelMutator based on ChannelMutator. It use
    OneShotMutableChannelGroup by default.

    Args:
        channel_group_cfg (Union[dict, Type[ChannelGroupType]], optional):
            Config of MutableChannelGroups. Defaults to
            dict( type='OneShotMutableChannelGroup',
            default_args=dict( num_blocks=8, min_blocks=2 ) ).
    """

    def __init__(self,
                 channel_group_cfg: Union[dict, Type[ChannelGroupType]] = dict(
                     type='OneShotMutableChannelGroup',
                     default_args=dict(num_blocks=8, min_blocks=2)),
                 **kwargs) -> None:

        super().__init__(channel_group_cfg, **kwargs)

    def min_choices(self) -> Dict:
        """Return the minimal pruning subnet(structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].min_choice
        return template

    def max_choices(self) -> Dict:
        """Return the maximal pruning subnet(structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].max_choice
        return template
