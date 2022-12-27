# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch

from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS
from .chex_unit import ChexUnit


@MODELS.register_module()
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
        with torch.no_grad():
            choices = self._get_prune_choices()
            for unit in self.mutable_units:
                unit.prune(choices[unit.name])

    def grow(self, growth_ratio=0.0):
        """Make the model grow.

        step1: get growth choices
        step2: grow based on ChexMixin.growth_imp
        """
        choices = self._get_grow_choices(growth_ratio)
        for unit in self.mutable_units:
            unit: ChexUnit
            unit.grow(choices[unit.name] - unit.current_choice)

    def _get_grow_choices(self, growth_ratio):
        choices = self._get_choices_by_bn_imp(self.channel_ratio +
                                              growth_ratio)

        return choices

    def _get_prune_choices(self):
        return self._get_choices_by_bn_imp(self.channel_ratio)

    def _get_choices_by_bn_imp(self, remain_ratio=0.5):

        choices = {}
        bn_imps = {}
        for unit in self.mutable_units:
            unit: ChexUnit
            bn_imps[unit.name] = unit.bn_imp
        bn_imp: torch.Tensor = torch.cat(list(bn_imps.values()), dim=0)

        num_total_channel = len(bn_imp)
        num_min_remained = int(remain_ratio * num_total_channel)
        threshold = bn_imp.topk(num_min_remained)[0][-1]

        num_remained = 0
        for unit in self.mutable_units:
            num = (bn_imps[unit.name] >= threshold).long().sum().item()
            choices[unit.name] = max(num, 1)
            num_remained += choices[unit.name]
        assert num_remained >= num_min_remained, \
            f'{num_remained},{num_min_remained}'
        return choices
