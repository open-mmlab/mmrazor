# Copyright (c) OpenMMLab. All rights reserved.
from typing import Type, Union

from torch.nn import Module

from ....registry import MODELS
from ...mutables.mutable_channel.groups.one_shot_channel_group import \
    OneShotChannelGroup
from .channel_mutator import MUTABLECHANNELGROUP, BaseChannelMutator


@MODELS.register_module()
class OneShotChannelMutator(BaseChannelMutator[OneShotChannelGroup]):

    def __init__(self,
                 model: Module,
                 channl_group_cfg: Union[dict,
                                         Type[MUTABLECHANNELGROUP]] = dict(
                                             type='OneShotChannelGroup',
                                             num_blocks=8,
                                             min_blocks=2),
                 **kwargs) -> None:
        super().__init__(model, channl_group_cfg, **kwargs)

    def min_structure(self):
        template = self.subnet_template()
        for key in template:
            template[key] = self._name2group[key].min_choice
        return template

    def max_structure(self):
        template = self.subnet_template()
        for key in template:
            template[key] = self._name2group[key].max_choice
        return template
