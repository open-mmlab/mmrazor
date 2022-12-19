# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.models.algorithms import BaseAlgorithm


class ChexAlgoritm(BaseAlgorithm):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 delta_t=2,
                 total_steps=10,
                 init_growth_rate=0.3,
                 init_cfg: Optional[Dict] = None):
        self.delta_t = delta_t
        self.init_growth_rate = init_growth_rate

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        if True:  #
            self.mutator.prune()
            self.mutator.grow(self.growth_ratio)
        return super().forward(inputs, data_samples, mode)

    @property
    def _epoch(self):
        pass

    @property
    def growth_ratio(self):
        # return growth ratio in current epoch
        pass
