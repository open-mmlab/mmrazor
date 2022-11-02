# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine import MessageHub, MMLogger
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models.mutables import MutableChannelUnit
from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


class ItePruneConfigManager:
    """ItePruneConfigManager manages the config of the structure of the model
    during pruning.

    Args:
        target (Dict[str, Union[int, float]]): The target structure to prune.
        supernet (Dict[str, Union[int, float]]): The sturecture of the
            supernet.
        epoch_step (int, optional): The prune step to prune. Defaults to 1.
        times (int, optional): The times to prune. Defaults to 1.
    """

    def __init__(self,
                 target: Dict[str, Union[int, float]],
                 supernet: Dict[str, Union[int, float]],
                 epoch_step=1,
                 times=1) -> None:

        self.supernet = supernet
        self.target = target
        self.epoch_step = epoch_step
        self.prune_times = times

        self.delta: Dict = self._get_delta_each_epoch(self.target,
                                                      self.supernet,
                                                      self.prune_times)

    def is_prune_time(self, epoch, ite):
        """Is the time to prune during training process."""
        return epoch % self.epoch_step == 0 \
            and epoch//self.epoch_step < self.prune_times \
            and ite == 0

    def prune_at(self, epoch):
        """Get the pruning structure in a time(epoch)."""
        times = epoch // self.epoch_step + 1
        assert times <= self.prune_times
        prune_current = {}
        ratio = times / self.prune_times

        for key in self.target:
            prune_current[key] = (self.target[key] - self.supernet[key]
                                  ) * ratio + self.supernet[key]
            if isinstance(self.supernet[key], int):
                prune_current[key] = int(prune_current[key])
        return prune_current

    def _get_delta_each_epoch(self, target: Dict, supernet: Dict, times: int):
        """Get the structure change for pruning once."""
        delta = {}
        for key in target:
            one_target = target[key]
            if isinstance(one_target, float):
                delta[key] = (1.0 - one_target) / times
            elif isinstance(one_target, int):
                delta[key] = int((supernet[key] - one_target) / times)
            else:
                raise NotImplementedError()
        return delta


@MODELS.register_module()
class ItePruneAlgorithm(BaseAlgorithm):
    """ItePruneAlgorithm prunes a model iteratively until reaching a prune-
    target.

    Args:
        architecture (Union[BaseModel, Dict]): The model to be pruned.
        mutator_cfg (Union[Dict, ChannelMutator], optional): The config
            of a mutator. Defaults to dict( type='ChannelMutator',
            channel_unit_cfg=dict( type='SequentialMutableChannelUnit')).
        data_preprocessor (Optional[Union[Dict, nn.Module]], optional):
            Defaults to None.
        target_pruning_ratio (dict, optional): The prune-target. The template
            of the prune-target can be get by calling
            mutator.choice_template(). Defaults to {}.
        step_epoch (int, optional): The step between two pruning operations.
            Defaults to 1.
        prune_times (int, optional): The times to prune a model. Defaults to 1.
        init_cfg (Optional[Dict], optional): init config for architecture.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator_cfg: Union[Dict, ChannelMutator] = dict(
                     type='ChannelMutator',
                     channel_unit_cfg=dict(
                         type='SequentialMutableChannelUnit')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 target_pruning_ratio: Optional[Dict[str, float]] = None,
                 step_epoch=1,
                 prune_times=1,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)

        # mutator
        self.mutator: ChannelMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

        if target_pruning_ratio is None:
            group_target_ratio = self.mutator.current_choices
        else:
            group_target_ratio = self.group_target_pruning_ratio(
                target_pruning_ratio, self.mutator.search_groups)

        # config_manager
        self.prune_config_manager = ItePruneConfigManager(
            group_target_ratio,
            self.mutator.current_choices,
            step_epoch,
            times=prune_times)

    def group_target_pruning_ratio(
        self, target: Dict[str, float],
        search_groups: Dict[int,
                            List[MutableChannelUnit]]) -> Dict[int, float]:
        """According to the target pruning ratio of each unit, set the target
        ratio of each search group."""
        group_target: Dict[int, float] = dict()
        for group_id, units in search_groups.items():
            for unit in units:
                unit_name = unit.name
                # The config of target pruning ratio does not
                # contain all units.
                if unit_name not in target:
                    continue
                if group_id in group_target:
                    unit_target = target[unit_name]
                    if unit_target != group_target[group_id]:
                        group_names = [u.name for u in units]
                        raise ValueError(
                            f"'{unit_name}' target ratio is different from "
                            f'other units in the same group {group_names}. '
                            'Pls check your target pruning ratio config.')
                else:
                    unit_target = target[unit_name]
                    assert isinstance(unit_target, (float, int))
                    group_target[group_id] = unit_target

        return group_target

    def check_prune_target(self, config: Dict):
        """Check if the prune-target is supported."""
        for value in config.values():
            assert isinstance(value, int) or isinstance(value, float)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward."""
        print(self._epoch, self._iteration)
        if self.prune_config_manager.is_prune_time(self._epoch,
                                                   self._iteration):

            config = self.prune_config_manager.prune_at(self._epoch)

            self.mutator.set_choices(config)

            logger = MMLogger.get_current_instance()
            logger.info(f'The model is pruned at {self._epoch}th epoch once.')

        return super().forward(inputs, data_samples, mode)

    def init_weights(self):
        return self.architecture.init_weights()

    # private methods

    @property
    def _epoch(self):
        """Get current epoch number."""
        message_hub = MessageHub.get_current_instance()
        if 'epoch' in message_hub.runtime_info:
            return message_hub.runtime_info['epoch']
        else:
            return 0

    @property
    def _iteration(self):
        """Get current iteration number."""
        message_hub = MessageHub.get_current_instance()
        if 'iter' in message_hub.runtime_info:
            iter = message_hub.runtime_info['iter']
            max_iter = message_hub.runtime_info['max_iters']
            max_epoch = message_hub.runtime_info['max_epochs']
            return iter % (max_iter // max_epoch)
        else:
            return 0
