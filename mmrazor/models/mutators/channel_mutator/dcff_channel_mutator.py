# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
from torch.nn import Module

from mmrazor.models.architectures.dynamic_op.bricks import DynamicChannelMixin
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
        print("len group_subnets:", group_subnets)
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
            print("group:", group.name, group)
            group: MUTABLECHANNELGROUP
            for channel in group.output_related:
                module2group[channel.name] = group

        return module2group

    def _convert_subnet(self, subnet: Dict[str, int]):
        group_subnets = {}
        print('module2group len:', len(self.module2group))
        for key in subnet:
            origin_key = key
            print(key)
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
        print("group_subnets:", group_subnets)
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
        # print("search_groups", self._search_groups[0])
        # print("len:", len(self._search_groups))
        print("group len:",len(self.groups))
        for layerid, group in enumerate(self.groups):
            if('FuseConv2d' not in group.name):
                continue
            print(group.name, type(group))
            print(group)
            print(group._model)
            param = group._model.weight

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
            softmaxp = self._clac_softmaxp(layeri_kl, layeri_softmaxp, layerid,
                                           cur_num, start_num,
                                           group.real_cout)
            # store updated state softmaxp to mutator's dict
            # print(type(mutable.layeri_softmaxp), type(mutable.layeri_softmaxp.data))
            group._model.layeri_softmaxp.data = softmaxp
            del param, layeri_param, layeri_negaEudist, layeri_kl

    def _clac_softmaxp(
        self,
        layeri_kl: torch.Tensor,
        layeri_softmaxp: torch.Tensor,
        layerid: int,
        cur_num: int,
        start_num: int,
        real_cout: int):
        layeri_iscore_kl = torch.sum(layeri_kl, dim=1)
        if cur_num == start_num and cur_num == 0:
            # first iter/epoch
            self.layers_iscore.append(layeri_iscore_kl)
        else:
            # no runner.meta to save layeri_iscore, so calculate kl on ckpt
            if False not in torch.isfinite(layeri_iscore_kl):
                # iter_runner cannot load meta in mmcv, calc instead
                if len(self.layers_iscore) <= layerid:
                    self.layers_iscore.append(layeri_iscore_kl)
                else:
                    self.layers_iscore[layerid] = layeri_iscore_kl
            else:
                pass

        # Gets the index value of the max k scores
        # in the layerid fusecov2d layer.
        _, topm_ids_order = torch.topk(
            self.layers_iscore[layerid], int(real_cout), sorted=False)

        softmaxp = layeri_softmaxp[topm_ids_order, :]
        return softmaxp
