# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmrazor.models.mutables import SlimmableChannelGroup
from mmrazor.registry import MODELS
from .base_channel_mutator import BaseChannelMutator


@MODELS.register_module()
class SlimmableChannelMutator(BaseChannelMutator[SlimmableChannelGroup]):
    """__init__ _summary_

    Args:
        channel_group_cfg (Dict): The config of ChannelGroups. Defaults to
            dict( type='SlimmableChannelGroup', groups={}).
        parse_cfg (Dict): The config of the tracer to parse the model.
            Defaults to dict( type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')).
        init_cfg (dict, optional): initialization configuration dict for
            BaseModule.
    """

    def __init__(self,
                 channel_group_cfg=dict(
                     type='SlimmableChannelGroup', groups={}),
                 parse_cfg=dict(
                     type='BackwardTracer',
                     loss_calculator=dict(type='ImageClassifierPseudoLoss')),
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(channel_group_cfg, parse_cfg, init_cfg)

        self.subnets = self._prepare_subnets(self.groups_cfg)

    # private methods

    def _prepare_subnets(self, group_cfg: Dict) -> List[Dict[str, int]]:
        """Prepare subnet config.

        Args:
            group_cfg (Dict[str, Dict[str]]): Config of the groups.
                group_cfg follows the below template:
                    {
                        'xx_group_name':{
                            'init_args':{
                                'candidate_choices':[c1,c2,c3...],...
                            },...
                        },...
                    }
                Every group must have the same number of candidate_choices, and
                the candidate in the list of candidate_choices with the same
                position compose a subnet.

        Returns:
            List[Dict[str, int]]: config of the subnets.
        """
        """Prepare subnet config."""
        subnets: List[Dict[str, int]] = []
        num_subnets = 0
        for key in group_cfg:
            num_subnets = len(group_cfg[key]['init_args']['candidate_choices'])
            break
        for _ in range(num_subnets):
            subnets.append({})
        for key in group_cfg:
            assert num_subnets == len(
                group_cfg[key]['init_args']['candidate_choices'])
            for i, value in enumerate(
                    group_cfg[key]['init_args']['candidate_choices']):
                subnets[i][key] = value

        return subnets
