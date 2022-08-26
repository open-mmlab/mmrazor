# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from torch.nn import Module

from mmrazor.models.mutables import SlimmableChannelGroup
from mmrazor.registry import MODELS
from .base_channel_mutator import MUTABLECHANNELGROUP, BaseChannelMutator


@MODELS.register_module()
class SlimmableChannelMutator(BaseChannelMutator[SlimmableChannelGroup]):

    def __init__(self,
                 channel_cfgs: Dict,
                 channl_group_cfg=dict(type='SlimmableChannelGroup'),
                 tracer_cfg=dict(
                     type='BackwardTracer',
                     loss_calculator=dict(type='ImageClassifierPseudoLoss')),
                 skip_prefixes: Optional[List[str]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(channl_group_cfg, tracer_cfg, skip_prefixes, init_cfg)
        self._subnets = self._prepare_subnets(channel_cfgs)

    def set_choices(self, config):
        config = self._convert_subnet(config)
        return super().set_choices(config)

    @property
    def subnets(self):
        return self._subnets

    # prepare model

    def prepare_from_supernet(self, supernet: Module) -> None:
        super().prepare_from_supernet(supernet)
        self.module2group = self._get_module2group()
        self._reset_group_candidates()

    # private methods

    def _reset_group_candidates(self):
        group_subnets = [
            self._convert_subnet(subnet) for subnet in self.subnets
        ]
        for key in group_subnets[0]:
            candidates = self._candidates_of(group_subnets, key)
            group: SlimmableChannelGroup = self._name2group[key]
            group.alter_candidates_after_init(candidates)

    def _prepare_subnets(self, channel_cfg: Dict[str, Dict[str, List[int]]]):
        subnets: List[Dict[str, int]] = []
        for key in channel_cfg:
            num_subnets = len(channel_cfg[key]['current_choice'])
            break
        for _ in range(num_subnets):
            subnets.append({})
        for key in channel_cfg:
            assert num_subnets == len(channel_cfg[key]['current_choice'])
            for i, value in enumerate(channel_cfg[key]['current_choice']):
                subnets[i][key] = value

        return subnets

    def _candidates_of(self, subnets, key):
        return [subnet[key] for subnet in subnets]

    def _get_module2group(self):
        module2group = dict()
        for group in self.groups:
            group: MUTABLECHANNELGROUP
            for channel in group.output_related:
                module2group[channel.name] = group

        return module2group

    def _convert_subnet(self, subnet: Dict[str, int]):
        group_subnets = {}
        for key in subnet:
            origin_key = key

            if 'mutable_out_channels' in key:
                key = key.replace('.mutable_out_channels', '')
            elif 'mutable_num_features' in key:
                key = key.replace('.mutable_num_features', '')
            else:
                continue

            if key in self.module2group:
                group = self.module2group[key]
                if group.name not in group_subnets:
                    group_subnets[group.name] = subnet[origin_key]
                else:
                    assert group_subnets[group.name] == subnet[origin_key]
            else:
                raise KeyError(f'{key} can not be found in module2group')
        return group_subnets
