# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .sequential_channel_group import SequentialChannelGroup


@MODELS.register_module()
class L1ChannelGroup(SequentialChannelGroup):

    @property
    def is_prunable(self):
        """bool: if the channel-group is prunable"""
        has_conv = False
        for module in self.output_related:
            if isinstance(module.module, nn.Conv2d):
                has_conv = True

        return super().is_prunable and has_conv

    # private methods

    def _generate_mask(self, choice: Union[int, float]) -> torch.Tensor:
        choice = self._get_int_choice(choice)
        norm = self._get_group_norm()
        idx = norm.topk(choice)[1]
        mask = torch.zeros([self.num_channels]).to(idx.device)
        mask.scatter_(0, idx, 1)
        return mask

    def _get_conv_norm(self, conv: nn.Conv2d, start, end):
        weight = conv.weight  # out_c * in_c * k * k
        weight = weight[start:end]
        norm = weight.abs().sum(dim=[1, 2, 3])
        return norm

    def _get_group_norm(self):
        avg_norm = 0
        conv_num = 0
        for channel in self.output_related:
            if isinstance(channel.module, nn.Conv2d):
                norm = self._get_conv_norm(channel.module, channel.start,
                                           channel.end)
                avg_norm += norm
                conv_num += 1
        avg_norm = avg_norm / conv_num
        return avg_norm
