# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Type, Union

from mmrazor.models.mutables import OneShotMutableChannelGroup
from mmrazor.registry import MODELS
from .base_channel_mutator import BaseChannelMutator, ChannelGroupType


@MODELS.register_module()
class OneShotChannelMutator(BaseChannelMutator[OneShotMutableChannelGroup]):
    """OneShotChannelMutator based on BaseChannelMutator. It use
    OneShotMutableChannelGroup by default.

    Args:
        channl_group_cfg (Union[dict, Type[ChannelGroupType]], optional):
            Config of MutableChannelGroups. Defaults to
            dict( type='OneShotMutableChannelGroup',
            default_args=dict( num_blocks=8, min_blocks=2 ) ).
    """

    def __init__(self,
                 channl_group_cfg: Union[dict, Type[ChannelGroupType]] = dict(
                     type='OneShotMutableChannelGroup',
                     default_args=dict(num_blocks=8, min_blocks=2)),
                 **kwargs) -> None:

        super().__init__(channl_group_cfg, **kwargs)

    def min_choices(self) -> Dict:
        """Return the minimal pruning subnet(structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].min_choice
        return template

    def max_choices(self):
        """Return the maximal pruning subnet(structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].max_choice
        return template
