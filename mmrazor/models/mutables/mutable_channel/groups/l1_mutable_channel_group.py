# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .sequential_mutable_channel_group import SequentialMutableChannelGroup


@MODELS.register_module()
class L1MutableChannelGroup(SequentialMutableChannelGroup):
    """Implementation of L1-norm pruning algorithm. It compute the l1-norm of
    modules and preferly prune the modules with less l1-norm.

    Please refer to papre `https://arxiv.org/pdf/1608.08710.pdf` for more
    detail.
    """

    def __init__(self,
                 num_channels: int,
                 choice_mode='number',
                 divisor=1,
                 min_value=1,
                 min_ratio=0.9) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio)

    # private methods

    def _generate_mask(self, choice: int) -> torch.Tensor:
        """Generate mask using choice."""
        norm = self._get_group_norm()
        idx = norm.topk(choice)[1]
        mask = torch.zeros([self.num_channels]).to(idx.device)
        mask.scatter_(0, idx, 1)
        return mask

    def _get_l1_norm(self, module: Union[nn.Conv2d, nn.Linear], start, end):
        """Get l1-norm of a module."""
        if isinstance(module, nn.Conv2d):
            weight = module.weight.flatten(1)  # out_c * in_c * k * k
        elif isinstance(module, nn.Linear):
            weight = module.weight  # out_c * in_c
        weight = weight[start:end]
        norm = weight.abs().mean(dim=[1])
        return norm

    def _get_group_norm(self):
        """Get l1-norm of the group by averaging the l1-norm of the moduls in
        the group."""
        avg_norm = 0
        module_num = 0
        for channel in self.output_related:
            if isinstance(channel.module, nn.Conv2d) or isinstance(
                    channel.module, nn.Linear):
                norm = self._get_l1_norm(channel.module, channel.start,
                                         channel.end)
                avg_norm += norm
                module_num += 1
        avg_norm = avg_norm / module_num
        return avg_norm
