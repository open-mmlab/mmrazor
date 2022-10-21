# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Type, Union

from mmrazor.models.architectures.dynamic_ops import FuseConv2d
from mmrazor.models.mutables import DCFFChannelUnit
from mmrazor.registry import MODELS
from .channel_mutator import ChannelMutator, ChannelUnitType


@MODELS.register_module()
class DCFFChannelMutator(ChannelMutator[DCFFChannelUnit]):
    """DCFF channel mutable based channel mutator. It uses DCFFChannelUnit.

    Args:
        channel_unit_cfg (Union[dict, Type[ChannelUnitType]], optional):
            Config of MutableChannelUnits. Defaults to
            dict( type='DCFFChannelUnit', units={}).
        parse_cfg (Dict): The config of the tracer to parse the model.
            Defaults to dict( type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')).
    """

    def __init__(self,
                 channel_unit_cfg: Union[dict, Type[ChannelUnitType]] = dict(
                     type='DCFFChannelUnit', units={}),
                 parse_cfg=dict(
                     type='BackwardTracer',
                     loss_calculator=dict(type='ImageClassifierPseudoLoss')),
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)
        self._channel_cfgs = channel_unit_cfg
        self._subnets = self._prepare_subnets(self.units_cfg)

    @property
    def subnets(self):
        return self._subnets

    def _prepare_subnets(self, unit_cfg: Dict) -> List[Dict[str, int]]:
        """Prepare subnet config.

        Args:
            unit_cfg (Dict[str, Dict[str]]): Config of the units.
                unit_cfg follows the below template:
                    {
                        'xx_unit_name':{
                            'init_args':{
                                'candidate_choices':[c],...
                            },...
                        },...
                    }
                Every unit must have the same number of candidate_choices, and
                the candidate in the list of candidate_choices with the same
                position compose a subnet.

        Returns:
            List[Dict[str, int]]: config of the subnets.
        """
        """Prepare subnet config."""
        subnets: List[Dict[str, int]] = []
        num_subnets = 0
        for key in unit_cfg:
            num_subnets = len(unit_cfg[key]['init_args']['candidate_choices'])
            break
        for _ in range(num_subnets):
            subnets.append({})
        for key in unit_cfg:
            assert num_subnets == len(
                unit_cfg[key]['init_args']['candidate_choices'])
            for i, value in enumerate(
                    unit_cfg[key]['init_args']['candidate_choices']):
                subnets[i][key] = value

        return subnets

    def calc_information(self, tau: float):
        """Calculate channel's kl and apply softmax pooling on channel to solve
        CUDA out of memory problem. KL calculation & pool are conducted in ops.

        Args:
            tau (float): temporature calculated by iter or epoch
        """
        # Calculate the filter importance of the current epoch.

        for layerid, unit in enumerate(self.units):
            for channel in unit.output_related:
                if isinstance(channel.module, FuseConv2d):
                    channel.module.get_pooled_channel(tau)
