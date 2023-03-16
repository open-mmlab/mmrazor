# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model.utils import _BatchNormXd

from mmrazor.models.mutables import (L1MutableChannelUnit,
                                     MutableChannelContainer)
from .ops import ExpandableBatchNorm2d, ExpandableConv2d, ExpandLinear


class ExpandableUnit(L1MutableChannelUnit):
    """The units to inplace modules with expandable dynamic ops."""

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ExpandableConv2d,
                nn.BatchNorm2d: ExpandableBatchNorm2d,
                _BatchNormXd: ExpandableBatchNorm2d,
                nn.Linear: ExpandLinear,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def expand(self, num):
        expand_mask = self.mutable_channel.mask.new_zeros([num])
        mask = torch.cat([self.mutable_channel.mask, expand_mask])
        self.mutable_channel.mask = mask

    def expand_to(self, num):
        self.expand(num - self.num_channels)
