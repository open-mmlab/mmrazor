# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine import MMLogger
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutators import DCFFChannelMutator
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
from .ite_prune_algorithm import ItePruneAlgorithm, ItePruneConfigManager

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class DCFF(ItePruneAlgorithm):
    """DCFF Networks.

    Please refer to paper
    [Dynamic-coded Filter Fusion](https://arxiv.org/abs/2107.06916).

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
            Defaults to 1. Legal input includes [1, self._max_iters]
            One and only one of (step_freq, prune_times) is set to legal int.
        prune_times (int, optional): The total times to prune a model.
            Defaults to 0. Legal input includes [1, self._max_iters]
            One and only one of (step_freq, prune_times) is set to legal int.
        init_cfg (Optional[Dict], optional): init config for architecture.
            Defaults to None.
        linear_schedule (bool, optional): flag to set linear ratio schedule.
            Defaults to False due to dcff fixed pruning rate.
        is_deployed (bool, optional): flag to set deployed algorithm.
            Defaults to False.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator_cfg: Union[Dict, DCFFChannelMutator] = dict(
                     type=' DCFFChannelMutator',
                     channel_unit_cfg=dict(type='DCFFChannelUnit')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 target_pruning_ratio: Optional[Dict[str, float]] = None,
                 step_freq=1,
                 prune_times=0,
                 init_cfg: Optional[Dict] = None,
                 linear_schedule=False,
                 is_deployed=False) -> None:
        # invalid param prune_times, reset after message_hub get [max_epoch]
        super().__init__(architecture, mutator_cfg, data_preprocessor,
                         target_pruning_ratio, step_freq, prune_times,
                         init_cfg, linear_schedule)
        self.is_deployed = is_deployed
        if (self.is_deployed):
            # To static ops for loaded pruned network.
            self._deploy()

    def _fix_archtecture(self):
        for module in self.architecture.modules():
            if isinstance(module, BaseMutable):
                if not module.is_fixed:
                    module.fix_chosen(None)

    def _deploy(self):
        config = self.prune_config_manager.prune_at(self._iter)
        self.mutator.set_choices(config)
        self.mutator.fix_channel_mutables()
        self._fix_archtecture()
        _dynamic_to_static(self.architecture)
        self.is_deployed = True

    def _calc_temperature(self, cur_num: int, max_num: int):
        """Calculate temperature param."""
        # Set the fixed parameters required to calculate the temperature t
        t_s, t_e, k = 1, 10000, 1

        A = 2 * (t_e - t_s) * (1 + math.exp(-k * max_num)) / (
            1 - math.exp(-k * max_num))
        T = A / (1 + math.exp(-k * cur_num)) + t_s - A / 2
        t = 1 / T
        return t

    def _legal_freq_time(self, freq_time):
        """check whether step_freq or prune_times belongs to legal range:

            [1, self._max_iters]

        Args:
            freq_time (Int): step_freq or prune_times.
        """
        return (freq_time > 0) and (freq_time < self._max_iters)

    def _init_prune_config_manager(self):
        """init prune_config_manager and check step_freq & prune_times.

        In DCFF, prune_times is set by step_freq and self._max_iters.
        """
        if self.target_pruning_ratio is None:
            group_target_ratio = self.mutator.current_choices
        else:
            group_target_ratio = self.group_target_pruning_ratio(
                self.target_pruning_ratio, self.mutator.search_groups)

        if self.by_epoch:
            # step_freq based on iterations
            self.step_freq *= self._iters_per_epoch

        if self._legal_freq_time(self.step_freq) ^ self._legal_freq_time(
                self.prune_times):
            if self._legal_freq_time(self.step_freq):
                self.prune_times = self._max_iters // self.step_freq
            else:
                self.step_freq = self._max_iters // self.prune_times
        else:
            raise RuntimeError('One and only one of (step_freq, prune_times)'
                               'can be set to legal int.')

        # config_manager move to forward.
        # message_hub['max_epoch'] unaccessible when init
        prune_config_manager = ItePruneConfigManager(
            group_target_ratio,
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
        # In DCFF prune_message is related to total_num
        # Set self.prune_config_manager after message_hub has['max_epoch/iter']
        if not hasattr(self, 'prune_config_manager'):
            # iter num per epoch only available after initiation
            self.prune_config_manager = self._init_prune_config_manager()
        if self.prune_config_manager.is_prune_time(self._iter):
            config = self.prune_config_manager.prune_at(self._iter)
            self.mutator.set_choices(config)

            # calc fusion channel
            temperature = self._calc_temperature(self._iter, self._max_iters)
            self.mutator.calc_information(temperature)

            logger = MMLogger.get_current_instance()
            if (self.by_epoch):
                logger.info(
                    f'The model is pruned at {self._epoch}th epoch once.')
            else:
                logger.info(
                    f'The model is pruned at {self._iter}th iter once.')

        return super().forward(inputs, data_samples, mode)
