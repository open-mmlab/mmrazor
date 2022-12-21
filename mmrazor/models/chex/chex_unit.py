# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.mutables.mutable_channel import MutableChannelContainer
from mmrazor.models.mutables.mutable_channel.units import L1MutableChannelUnit
from mmrazor.registry import MODELS
from .chex_ops import ChexConv2d, ChexLinear, ChexMixin


@MODELS.register_module()
class ChexUnit(L1MutableChannelUnit):

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ChexConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: ChexLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                _BatchNormXd: dynamic_ops.DynamicBatchNormXd,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def prune(self, num_remaining):
        # prune the channels to num_remaining
        def get_prune_imp():
            prune_imp = 0
            for channel in self.chex_channels:
                module = channel.module
                prune_imp = prune_imp + module.prune_imp(
                    num_remaining)[channel.start:channel.end]
            return prune_imp

        with torch.no_grad():
            prune_imp = get_prune_imp()
            index = prune_imp.topk(num_remaining)[1]
            self.mutable_channel.mask.fill_(0.0)
            self.mutable_channel.mask.data.scatter_(-1, index, 1.0)

    def grow(self, num):
        assert num >= 0
        if num == 0:
            return

        def get_growth_imp():
            growth_imp = 0
            for channel in self.chex_channels:
                module = channel.module
                growth_imp = growth_imp + module.growth_imp[channel.
                                                            start:channel.end]
            return growth_imp

        growth_imp = get_growth_imp()
        mask = self.mutable_channel.current_mask
        index_free = torch.nonzero(1 - mask.float()).flatten()
        growth_imp = growth_imp[index_free]
        growth_imp = growth_imp.softmax(dim=-1)
        if len(index_free) >= num:
            select_index = torch.multinomial(growth_imp, num)
            select_index = index_free[select_index]
        else:
            select_index = index_free

        self.mutable_channel.mask.index_fill_(-1, select_index, 1.0)

    @property
    def bn_imp(self):
        imp = 0
        num_layers = 0
        for channel in self.output_related:
            module = channel.module
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                imp = imp + module.weight[channel.start:channel.end]
                num_layers += 1
        assert num_layers > 0
        imp = imp / num_layers
        return imp

    @property
    def chex_channels(self):
        for channel in self.output_related:
            module = channel.module
            if isinstance(module, ChexMixin):
                yield channel
