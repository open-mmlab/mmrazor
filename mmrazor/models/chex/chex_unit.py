# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch.nn as nn

from mmrazor.models.mutables.mutable_channel.units import \
    SequentialMutableChannelUnit


class ChexUnit(SequentialMutableChannelUnit):

    def prepare_for_pruning(self, model: nn.Module):
        return super().prepare_for_pruning(model)

    @property
    def current_choice(self) -> Union[int, float]:
        return super().current_choice

    @current_choice.setter
    def current_choice(self, value):
        if self.current_choice > value:
            # prune
            self.prune(1)
        else:
            # growth
            self.prune(1)

    def grow(self, num):
        pass

    def prune(self, num):
        pass

    def bn_imp(self):
        # return channel importance based on bn
        pass
