# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine import MessageHub
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models.mutators import BaseChannelMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


class PruneConfigManager:

    def __init__(self,
                 prune_target: Dict[str, float],
                 prune_origin,
                 epoch_step=1,
                 times=1) -> None:
        self.prune_all = prune_origin
        self.prune_target = prune_target
        self.prune_delta: Dict = {}
        self.epoch_step = epoch_step
        self.prune_times = times

        self.get_prune_detla_each()

    def get_prune_detla_each(self):
        delta = {}
        for key in self.prune_target:
            target = self.prune_target[key]
            if isinstance(target, float):
                delta[key] = (1 - target) / self.prune_times
            elif isinstance(target, int):
                delta[key] = int(
                    (self.prune_all[key] - target) / self.prune_times)
            else:
                raise NotImplementedError()
        self.prune_delta = delta

    def is_prune_time(self, epoch, ite):
        return epoch % self.epoch_step == 0 \
            and epoch//self.epoch_step < self.prune_times \
            and ite == 0

    def prune_at(self, epoch):
        times = epoch // self.epoch_step + 1
        prune_current = {}
        if times == self.prune_times:
            return self.prune_target
        else:
            for key in self.prune_delta:
                prune_current[
                    key] = self.prune_all[key] - self.prune_delta[key] * times
            return prune_current


@MODELS.register_module()
class ItePruneAlgorithm(BaseAlgorithm):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator_cfg: Union[Dict, BaseChannelMutator] = dict(
                     type='ChannelMutator',
                     channl_group_cfg=dict(type='L1ChannelGroup')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 target_pruning_ratio={},
                 step_epoch=1,
                 prune_times=1,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        # mutator
        self.mutator: BaseChannelMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

        # config_manager
        self.check_prune_targe(target_pruning_ratio)
        self.prune_config_manager = PruneConfigManager(
            target_pruning_ratio,
            self.mutator.choice_template,
            step_epoch,
            times=prune_times)

    def check_prune_targe(self, config: Dict):
        for value in config.values():
            assert isinstance(value, int) or isinstance(value, float)

    @property
    def _epoch(self):
        message_hub = MessageHub.get_current_instance()
        if 'epoch' in message_hub.runtime_info:
            return message_hub.runtime_info['epoch']
        else:
            return 0

    @property
    def _iteration(self):
        message_hub = MessageHub.get_current_instance()
        if 'iter' in message_hub.runtime_info:
            return message_hub.runtime_info['iter']
        else:
            return 0

    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        if self.prune_config_manager.is_prune_time(self._epoch,
                                                   self._iteration):
            config = self.prune_config_manager.prune_at(self._epoch)
            self.mutator.set_choices(config)

        return super().forward(batch_inputs, data_samples, mode)

    def init_weights(self):
        return self.architecture.init_weights()
