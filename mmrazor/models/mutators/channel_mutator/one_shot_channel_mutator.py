# Copyright (c) OpenMMLab. All rights reserved.
from typing import Type, Union

from mmrazor.models.mutables import OneShotChannelGroup
from mmrazor.registry import MODELS
from .base_channel_mutator import MUTABLECHANNELGROUP, BaseChannelMutator


@MODELS.register_module()
class OneShotChannelMutator(BaseChannelMutator[OneShotChannelGroup]):

    def __init__(self,
                 channl_group_cfg: Union[dict,
                                         Type[MUTABLECHANNELGROUP]] = dict(
                                             type='OneShotChannelGroup',
                                             num_blocks=8,
                                             min_blocks=2),
                 **kwargs) -> None:
        super().__init__(channl_group_cfg, **kwargs)

    def min_choices(self):
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].min_choice
        return template

    def max_choices(self):
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].max_choice
        return template
