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
        step_freq (int, optional): The prune step of epoch/iter to prune.
            Defaults to 1.
        prune_times (int, optional): The times to prune. Defaults to 1.
        linear_schedule (bool, optional): flag to set linear ratio schedule.
            Defaults to True.
    """

    def __init__(self,
                 target: Dict[str, Union[int, float]],
                 supernet: Dict[str, Union[int, float]],
                 step_freq=1,
                 prune_times=1,
                 linear_schedule=True) -> None:

        self.supernet = supernet
        self.target = target
        self.step_freq = step_freq
        self.prune_times = prune_times
        self.linear_schedule = linear_schedule

        self.delta: Dict = self._get_delta_each_iter(self.target,
                                                     self.supernet,
                                                     self.prune_times)

    def is_prune_time(self, iteration):
        """Is the time to prune during training process."""
        return iteration % self.step_freq == 0 \
            and iteration // self.step_freq < self.prune_times

    def prune_at(self, iteration):
        """Get the pruning structure in a time(iteration)."""
        times = iteration // self.step_freq + 1
        assert times <= self.prune_times
        prune_current = {}
        ratio = times / self.prune_times

        for key in self.target:
            if self.linear_schedule:
                # TO DO: add scheduler for more pruning rate schedule
                prune_current[key] = (self.target[key] - self.supernet[key]
                                      ) * ratio + self.supernet[key]
            else:
                prune_current[key] = self.target[key]
            if isinstance(self.supernet[key], int):
                prune_current[key] = int(prune_current[key])
        return prune_current

    def _get_delta_each_iter(self, target: Dict, supernet: Dict, times: int):
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
        step_freq (int, optional): The step between two pruning operations.
            Defaults to 1.
        prune_times (int, optional): The total times to prune a model.
            Defaults to 1.
        init_cfg (Optional[Dict], optional): init config for architecture.
            Defaults to None.
        linear_schedule (bool, optional): flag to set linear ratio schedule.
            Defaults to True.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator_cfg: Union[Dict, ChannelMutator] = dict(
                     type='ChannelMutator',
                     channel_unit_cfg=dict(
                         type='SequentialMutableChannelUnit')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 target_pruning_ratio: Optional[Dict[str, float]] = None,
                 step_freq=1,
                 prune_times=1,
                 init_cfg: Optional[Dict] = None,
                 linear_schedule=True) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)

        # decided by EpochBasedRunner or IterBasedRunner
        self.target_pruning_ratio = target_pruning_ratio
        self.step_freq = step_freq
        self.prune_times = prune_times
        self.linear_schedule = linear_schedule

        self.mutator: ChannelMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

    def set_target_pruning_ratio(
            self, target: Dict[str, float],
            units: List[MutableChannelUnit]) -> Dict[str, float]:
        """According to the target pruning ratio of each unit, set the target
        ratio of each unit in units."""
        target_pruning_ratio: Dict[str, float] = dict()
        for unit in units:
            assert isinstance(unit, MutableChannelUnit), (
                f'unit should be `MutableChannelUnit`, but got {type(unit)}.')
            unit_name = unit.name
            # The config of target pruning ratio does not
            # contain all units.
            if unit_name not in target:
                continue
            unit_target = target[unit_name]
            assert isinstance(unit_target, (float, int))
            target_pruning_ratio[unit_name] = unit_target
        return target_pruning_ratio

    def check_prune_target(self, config: Dict):
        """Check if the prune-target is supported."""
        for value in config.values():
            assert isinstance(value, int) or isinstance(value, float)

    def _init_prune_config_manager(self):
        """init prune_config_manager and check step_freq & prune_times.

        message_hub['max_epoch/iter'] unaccessible when initiation.
        """
        if self.target_pruning_ratio is None:
            target_pruning_ratio = self.mutator.current_choices
        else:
            target_pruning_ratio = self.set_target_pruning_ratio(
                self.target_pruning_ratio, self.mutator.mutable_units)

        if self.by_epoch:
            # step_freq based on iterations
            self.step_freq *= self._iters_per_epoch

        # config_manager move to forward.
        # message_hub['max_epoch'] unaccessible when init
        prune_config_manager = ItePruneConfigManager(
            target_pruning_ratio,
            self.mutator.current_choices,
            self.step_freq,
            prune_times=self.prune_times,
            linear_schedule=self.linear_schedule)

        return prune_config_manager

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward."""

        if self.training:
            if not hasattr(self, 'prune_config_manager'):
                # self._iters_per_epoch() only available after initiation
                self.prune_config_manager = self._init_prune_config_manager()
            if self.prune_config_manager.is_prune_time(self._iter):

                config = self.prune_config_manager.prune_at(self._iter)

                self.mutator.set_choices(config)

                logger = MMLogger.get_current_instance()
                if (self.by_epoch):
                    logger.info(
                        f'The model is pruned at {self._epoch}th epoch once.')
                else:
                    logger.info(
                        f'The model is pruned at {self._iter}th iter once.')

        return super().forward(inputs, data_samples, mode)

    def init_weights(self):
        return self.architecture.init_weights()

    # private methods

    @property
    def by_epoch(self):
        """Get epoch/iter based train loop."""
        # IterBasedTrainLoop max_epochs default to 1
        # TO DO: Add by_epoch params or change default max_epochs?
        return self._max_epochs != 1

    @property
    def _epoch(self):
        """Get current epoch number."""
        message_hub = MessageHub.get_current_instance()
        if 'epoch' in message_hub.runtime_info:
            return message_hub.runtime_info['epoch']
        else:
            raise RuntimeError('Use MessageHub before initiation.'
                               'epoch is inited in before_run_epoch().')

    @property
    def _iter(self):
        """Get current sum iteration number."""
        message_hub = MessageHub.get_current_instance()
        if 'iter' in message_hub.runtime_info:
            return message_hub.runtime_info['iter']
        else:
            raise RuntimeError('Use MessageHub before initiation.'
                               'iter is inited in before_run_iter().')

    @property
    def _max_epochs(self):
        """Get max epoch number.

        Default 1 for IterTrainLoop
        """
        message_hub = MessageHub.get_current_instance()
        if 'max_epochs' in message_hub.runtime_info:
            return message_hub.runtime_info['max_epochs']
        else:
            raise RuntimeError('Use MessageHub before initiation.'
                               'max_epochs is inited in before_run_epoch().')

    @property
    def _max_iters(self):
        """Get max iteration number."""
        message_hub = MessageHub.get_current_instance()
        if 'max_iters' in message_hub.runtime_info:
            return message_hub.runtime_info['max_iters']
        else:
            raise RuntimeError('Use MessageHub before initiation.'
                               'max_iters is inited in before_run_iter().')

    @property
    def _iters_per_epoch(self):
        """Get iter num per epoch."""
        return self._max_iters / self._max_epochs
