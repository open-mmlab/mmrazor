# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
from torch.nn import Module

from mmrazor.models.architectures.dynamic_ops.bricks import FuseConv2d
from mmrazor.models.mutables import DCFFChannelGroup
from mmrazor.registry import MODELS
from .base_channel_mutator import BaseChannelMutator, ChannelGroupType


@MODELS.register_module()
class DCFFChannelMutator(BaseChannelMutator[DCFFChannelGroup]):
    """DCFF channel mutable based channel mutator. It uses DCFFChannelGroup.

    Args:
        channel_cfgs (list[Dict]): A list of candidate channel configs.
        channel_group_cfg (Union[dict, Type[ChannelGroupType]], optional):
            Config of MutableChannelGroups. Defaults to
            dict( type='DCFFChannelGroup').
    """

    def __init__(self,
                 channel_cfgs: Dict,
                 channl_group_cfg: Union[dict, Type[ChannelGroupType]] = dict(
                     type='DCFFChannelGroup'),
                 **kwargs) -> None:
        super().__init__(channl_group_cfg, **kwargs)
        self._channel_cfgs = channel_cfgs
        self._subnets = self._prepare_subnets(channel_cfgs)

    @property
    def subnets(self):
        return self._subnets

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Note:
            Different from `ChannelMutator`, we only support Case 1 in
            `ChannelMutator`. The input supernet should be made up of original
            nn.Module. And we replace the conv/linear/bn modules in the input
            supernet with dynamic ops first. Then we trace the topology of
            the supernet to get the `concat_parent_mutables` of a certain
            mutable, if the input of a module is a concatenation of several
            modules' outputs. Then we convert the ``DynamicBatchNorm`` in
            supernet with ``SwitchableBatchNorm2d``, and set the candidate
            channel numbers to the corresponding `SlimmableChannelMutable`.
            Finally, we establish the relationship between the current nodes
            and their parents.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        super().prepare_from_supernet(supernet)
        self.module2group = self._get_module2group()
        self._reset_group_candidates()

    def _reset_group_candidates(self):
        """Alter candidates of DCFFChannelGroup according to channel_cfgs."""
        # print("name2group:", self._name2group)
        for key in self._channel_cfgs:
            group: DCFFChannelGroup = self._name2group[key]
            group.alter_candidates_after_init(
                self._channel_cfgs[key]['candidates'])

    def _prepare_subnets(self, channel_cfg: Dict[str, Dict[str, List[int]]]):
        subnets: List[Dict[str, int]] = []
        num_subnets = 0
        for key in channel_cfg:
            num_subnets = len(channel_cfg[key]['candidates'])
            break
        for _ in range(num_subnets):
            subnets.append({})
        for key in channel_cfg:
            assert num_subnets == len(channel_cfg[key]['candidates'])
            for i, value in enumerate(channel_cfg[key]['candidates']):
                subnets[i][key] = value

        return subnets

    def _candidates_of(self, subnets, key):
        return [subnet[key] for subnet in subnets]

    def _get_module2group(self):
        module2group = dict()
        for group in self.groups:
            group: ChannelGroupType
            for channel in group.output_related:
                module2group[channel.name] = group

        return module2group

    def calc_information(self, tau: float):
        """calculate channel's kl and apply softmax pooling on channel to solve
        CUDA out of memory problem.

        Args:
            tau (float): temporature calculated by iter or epoch
        """
        # Calculate the filter importance of the current epoch.

        for layerid, group in enumerate(self.groups):
            for channel in group.output_related:
                if isinstance(channel.module, FuseConv2d):
                    param = channel.module.weight

                    # Compute layeri_param.
                    layeri_param = torch.reshape(param.detach(),
                                                 (param.shape[0], -1))
                    layeri_Eudist = torch.cdist(
                        layeri_param, layeri_param, p=2)
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
                    real_out = channel.module.mutable_attrs[
                        'out_channels'].activated_channels

                    layeri_iscore_kl = torch.sum(layeri_kl, dim=1)
                    _, topm_ids_order = torch.topk(
                        layeri_iscore_kl, int(real_out), sorted=False)
                    softmaxp = layeri_softmaxp[topm_ids_order, :]
                    if (not hasattr(channel.module, 'layeri_softmaxp')):
                        setattr(channel.module, 'layeri_softmaxp', softmaxp)
                    else:
                        channel.module.layeri_softmaxp.data = softmaxp
                    del param, layeri_param, layeri_negaEudist, layeri_kl
