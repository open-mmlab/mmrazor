# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
from torch.nn import Module

from mmrazor.models.architectures.dynamic_ops.bricks import FuseConv2d
from mmrazor.models.mutables import DCFFChannelGroup
from mmrazor.registry import MODELS
from mmrazor.structures.graph import ModuleGraph
from .base_channel_mutator import MUTABLECHANNELGROUP, BaseChannelMutator


@MODELS.register_module()
class DCFFChannelMutator(BaseChannelMutator[DCFFChannelGroup]):

    def __init__(self,
                 channel_cfgs: Dict,
                 channl_group_cfg: Union[dict,
                                         Type[MUTABLECHANNELGROUP]] = dict(
                                             type='DCFFChannelGroup'),
                 **kwargs) -> None:
        super().__init__(channl_group_cfg, **kwargs)
        self._subnets = self._prepare_subnets(channel_cfgs)

    def set_choices(self, config):
        config = self._convert_subnet(config)
        return super().set_choices(config)

    @property
    def subnets(self):
        return self._subnets

    def prepare_from_supernet(self, supernet: Module) -> None:
        super().prepare_from_supernet(supernet)
        self.module2group = self._get_module2group()
        self._reset_group_candidates()

    def _reset_group_candidates(self):
        group_subnets = [
            self._convert_subnet(subnet) for subnet in self.subnets
        ]
        for key in group_subnets[0]:
            candidates = self._candidates_of(group_subnets, key)
            group: DCFFChannelGroup = self._name2group[key]
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

    def calc_information(
        self,
        tau: float,
        cur_num: int,
        start_num: int):
        """calculate channel's kl and apply softmax pooling on channel solve
        CUDA out of memory.

        Args:
            t (float): temporature calculated by iter or epoch
            cur_num (int): current iter or epoch, used for resume
            start_num (int): start iter or epoch, used for resume
        """
        # Calculate the filter importance of the current epoch.
        for layerid, group in enumerate(self.groups):
            for channel in group.output_related:
                if isinstance(channel.module, FuseConv2d):
                    param = channel.module.weight

                    # Compute layeri_param.
                    layeri_param = torch.reshape(param.detach(), (param.shape[0], -1))
                    layeri_Eudist = torch.cdist(layeri_param, layeri_param, p=2)
                    layeri_negaEudist = -layeri_Eudist
                    softmax = nn.Softmax(dim=1)
                    layeri_softmaxp = softmax(layeri_negaEudist / tau)

                    # KL = [c, 1, c] * ([c, 1 ,c] / [c, c, 1]).log()
                    #    = [c, 1, c] * ([c, 1, c].log() - [c, c, 1].log())
                    # only dim0 is required, dim1 and dim2 are pooled
                    # calc mean(dim=1) first

                    # avoid frequent NaN
                    eps = 1e-7
                    layeri_kl = layeri_softmaxp[:, None, :]
                    log_p = layeri_kl * (layeri_kl + eps).log()
                    log_q = layeri_kl * torch.mean(
                        (layeri_softmaxp + eps).log(), dim=1)

                    layeri_kl = torch.mean((log_p - log_q), dim=2)
                    del log_p, log_q
                    real_out = channel.module.mutable_attrs['out_channels'].activated_channels

                    layeri_iscore_kl = torch.sum(layeri_kl, dim=1)
                    _, topm_ids_order = torch.topk(
                        layeri_iscore_kl, int(real_out), sorted=False)
                    softmaxp = layeri_softmaxp[topm_ids_order, :]
                    # softmaxp = self._clac_softmaxp(layeri_kl, layeri_softmaxp,
                    #                             cur_num, start_num, real_out)
                    # store updated state softmaxp to mutator's dict
                    if(not hasattr(channel.module, 'layeri_softmaxp')):
                        setattr(channel.module, 'layeri_softmaxp', softmaxp)
                    else:
                        channel.module.layeri_softmaxp.data = softmaxp
                    del param, layeri_param, layeri_negaEudist, layeri_kl

    def _clac_softmaxp(
        self,
        layeri_kl: torch.Tensor,
        layeri_softmaxp: torch.Tensor,
        # layerid: int,
        cur_num: int,
        start_num: int,
        real_cout: int):
        layeri_iscore_kl = torch.sum(layeri_kl, dim=1)
        # Gets the index value of the max k scores
        # in the layerid fusecov2d layer.
        _, topm_ids_order = torch.topk(
            layeri_iscore_kl, int(real_cout), sorted=False)

        softmaxp = layeri_softmaxp[topm_ids_order, :]
        return softmaxp
