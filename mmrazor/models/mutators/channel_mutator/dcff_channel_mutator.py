# Copyright (c) OpenMMLab. All rights reserved.
from typing import Type, Union

import torch
import torch.nn as nn
from torch.nn import Module

from mmrazor.models.architectures.dynamic_op.bricks import DynamicChannelMixin
from mmrazor.models.mutables import OneShotChannelGroup
from mmrazor.registry import MODELS
from mmrazor.structures.graph import ModuleGraph
from .base_channel_mutator import MUTABLECHANNELGROUP, BaseChannelMutator


@MODELS.register_module()
class DCFFChannelMutator(BaseChannelMutator[OneShotChannelGroup]):

    def __init__(self,
                 channl_group_cfg: Union[dict,
                                         Type[MUTABLECHANNELGROUP]] = dict(
                                             type='DCFFChannelGroup',
                                             num_blocks=8,
                                             min_blocks=2),
                 **kwargs) -> None:
        super().__init__(channl_group_cfg, **kwargs)

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Convert modules to dynamicops and parse channel groups."""

        # self.convert_dynamic_module(supernet, self.module_converters)
        supernet.eval()

        self.group_class.prepare_model(supernet)
        self._name2module = dict(supernet.named_modules())
        print(supernet)

        if self.tracer_cfg['type'] == 'BackwardTracer':
            graph = ModuleGraph.init_using_backward_tracer(
                supernet, self.tracer_cfg)
        elif self.tracer_cfg['type'] == 'fx':

            def is_dynamic_op_for_fx_tracer(module, module_name):
                """determine if a module is a dynamic op for fx tracer."""
                return isinstance(module, DynamicChannelMixin)

            graph = ModuleGraph.init_using_fx_tracer(
                supernet, is_dynamic_op_for_fx_tracer)
        else:
            raise NotImplementedError()

        print(graph)
        self._graph = graph
        self.groups = self.group_class.parse_channel_groups(
            graph, self.group_args)
        for group in self.groups:
            group.prepare_for_pruning()
            self._name2group[group.name] = group

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
