# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)


class ChexMixin:

    @property
    def prune_imp(self):
        # compute channel importance for pruning
        return self._prune_imp(self.weight)

    @property
    def growth_imp(self):
        # compute channel importance for growth
        return self._growth_imp(self.weight)

    def _prune_imp(self, weight):
        # weight: out * in. return the importance of each channel
        out_channel = weight.shape[0]
        return torch.rand(out_channel)

    def _growth_imp(self, weight):
        # weight: out * in. return the importance of each channel when growth
        out_channel = weight.shape[0]
        return torch.rand(out_channel)


class ChexConv2d(DynamicConv2d, ChexMixin):
    pass


class ChexLinear(DynamicLinear, ChexMixin):
    pass
