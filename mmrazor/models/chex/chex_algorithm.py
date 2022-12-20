# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Union

import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.registry import MODELS
from .chex_mutator import ChexMutator
from .utils import RuntimeInfo


@MODELS.register_module()
class ChexAlgorithm(BaseAlgorithm):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 mutator_cfg=dict(
                     type='ChexMutator',
                     channel_unit_cfg=dict(type='ChexUnit')),
                 delta_t=2,
                 total_steps=10,
                 init_growth_rate=0.3,
                 init_cfg: Optional[Dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.delta_t = delta_t
        self.total_steps = total_steps
        self.init_growth_rate = init_growth_rate

        self.mutator: ChexMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        if self.training:  #
            if RuntimeInfo.iter() % self.delta_t == 0 and \
                 RuntimeInfo.epoch() < self.total_steps:
                self.mutator.prune()
                self.mutator.grow(self.growth_ratio)
        return super().forward(inputs, data_samples, mode)

    @property
    def growth_ratio(self):
        # return growth ratio in current epoch
        def cos():
            a = math.pi * RuntimeInfo.epoch() / RuntimeInfo.max_epochs()
            return (math.cos(a) + 1) / 2

        return self.init_growth_rate * cos()
