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
from .ite_prune_algorithm import ItePruneAlgorithm

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
        mutator_cfg (Union[Dict, DCFFChannelMutator], optional): The config
            of a mutator. Defaults to dict( type='DCFFChannelMutator',
            channel_unit_cfg=dict( type='DCFFChannelUnit')).
        data_preprocessor (Optional[Union[Dict, nn.Module]], optional):
            Defaults to None.
        target_pruning_ratio (dict, optional): The prune-target. The template
            of the prune-target can be get by calling
            mutator.choice_template(). Defaults to {}.
        step_freq (int, optional): The step between two pruning operations.
            Defaults to 1.
        prune_times (int, optional): The times to prune a model. Defaults to 1.
        init_cfg (Optional[Dict], optional): init config for architecture.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator_cfg: Union[Dict, DCFFChannelMutator] = dict(
                     type=' DCFFChannelMutator',
                     channel_unit_cfg=dict(type='DCFFChannelUnit')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 target_pruning_ratio: Optional[Dict[str, float]] = None,
                 step_freq=1,
                 prune_times=1,
                 init_cfg: Optional[Dict] = None,
                 by_epoch=True,
                 is_deployed=False) -> None:
        super().__init__(architecture, mutator_cfg, data_preprocessor,
                         target_pruning_ratio, step_freq, prune_times,
                         init_cfg, by_epoch)
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
        config = self.prune_config_manager.prune_at(self._num)
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

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward."""
        if self.prune_config_manager.is_prune_time(self._num,
                                                   self._current_iteration):
            # calc fusion channel
            temperature = self._calc_temperature(self._num, self._max_num)
            self.mutator.calc_information(temperature)

            config = self.prune_config_manager.prune_at(self._num)
            self.mutator.set_choices(config)

            logger = MMLogger.get_current_instance()
            if (self.by_epoch):
                logger.info(
                    f'The model is pruned at {self._num}th epoch once.')
            else:
                logger.info(f'The model is pruned at {self._num}th iter once.')

        return super().forward(inputs, data_samples, mode)
