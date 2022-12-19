# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmrazor.models.mutators import ChannelMutator


class ChexMutator(ChannelMutator):

    def __init__(self,
                 channel_unit_cfg={},
                 parse_cfg: Dict = dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='BackwardTracer'),
                 custom_groups: Optional[List[List[str]]] = None,
                 channel_ratio=0.7,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, custom_groups, init_cfg)
        self.channel_ratio = channel_ratio  # number of channels to preserve

    def prune(self):
        """Prune the model.

        step1: get pruning structure
        step2: prune based on ChexMixin.prune_imp
        """
        _ = self._get_prune_choices()
        pass

    def grow(self, growth_ratio=0.0):
        """Make the model grow.

        step1: get growth choices
        step2: grow based on ChexMixin.growth_imp
        """
        _ = self._get_grow_choices(growth_ratio)
        pass

    def _get_grow_choices(self, growth_choice):
        pass

    def _get_prune_choices(self):
        pass
