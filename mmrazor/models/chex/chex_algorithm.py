# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from mmengine import dist
from mmengine.model import BaseModel
from mmengine.model.utils import convert_sync_batchnorm

from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.registry import MODELS
from mmrazor.utils import print_log
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

        if dist.is_distributed():
            self.architecture = convert_sync_batchnorm(self.architecture)

        self.delta_t = delta_t
        self.total_steps = total_steps
        self.init_growth_rate = init_growth_rate

        self.mutator: ChexMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        if self.training:  #
            if RuntimeInfo.epoch() % self.delta_t == 0 and \
                 RuntimeInfo.epoch() < self.total_steps and \
                    RuntimeInfo.iter_by_epoch() == 0:
                with torch.no_grad():
                    self.mutator.prune()
                    print_log(f'prune model with {self.mutator.channel_ratio}')
                    self.log_choices()

                    self.mutator.grow(self.growth_ratio)
                    print_log(f'grow model with {self.growth_ratio}')
                    self.log_choices()
        return super().forward(inputs, data_samples, mode)

    @property
    def growth_ratio(self):
        # return growth ratio in current epoch
        def cos():
            a = math.pi * RuntimeInfo.epoch() / self.total_steps
            return (math.cos(a) + 1) / 2

        return self.init_growth_rate * cos()

    def log_choices(self):
        if dist.get_rank() == 0:
            config = {}
            for unit in self.mutator.mutable_units:
                config[unit.name] = unit.current_choice
            print_log(json.dumps(config, indent=4))
