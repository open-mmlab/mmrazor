# Copyright (c) OpenMMLab. All rights reserved.
import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from mmengine import fileio
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutators import DCFFChannelMutator
from mmrazor.models.utils import reinitialize_optim_wrapper_count_status
from mmrazor.registry import MODELS
# from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[DCFFChannelMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]
VALID_PATH_TYPE = Union[str, Path]
VALID_CHANNEL_CFG_PATH_TYPE = Union[VALID_PATH_TYPE, List[VALID_PATH_TYPE]]


@MODELS.register_module()
class DCFF(BaseAlgorithm):

    def __init__(self,
                 mutator: VALID_MUTATOR_TYPE,
                 architecture: Union[BaseModel, Dict],
                 channel_cfgs: Union[str, Dict],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 fuse_count: int = 1,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        if isinstance(channel_cfgs, str):
            channel_cfgs = fileio.load(channel_cfgs)

        self.mutator = self._build_mutator(copy.copy(mutator), channel_cfgs)
        self.mutator.prepare_from_supernet(self.architecture)
        self.num_subnet = len(self.mutator.subnets)
        print('num_subnet:', self.num_subnet)
        self.fuse_count = fuse_count

        self._optim_wrapper_count_status_reinitialized = False

    def _fix_archtecture(self):
        for module in self.architecture.modules():
            if isinstance(module, BaseMutable):
                if not module.is_fixed:
                    module.fix_chosen(None)

    def _calc_temperature(self, cur_num: int, max_num: int):
        """Calculate temperature param."""
        # Set the fixed parameters required to calculate the temperature t
        t_s, t_e, k = 1, 10000, 1

        A = 2 * (t_e - t_s) * (1 + torch.math.exp(-k * max_num)) / (
            1 - torch.math.exp(-k * max_num))
        T = A / (1 + torch.math.exp(-k * cur_num)) + t_s - A / 2
        t = 1 / T
        return t

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Train step."""

        # self.message_hub = MessageHub.get_current_instance()
        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=self.num_subnet)
            self._optim_wrapper_count_status_reinitialized = True
        self.message_hub = optim_wrapper.message_hub.get_current_instance()
        self.max_num = self.message_hub._runtime_info['max_epochs']
        # buffer not available in __init__()
        self.by_epoch = (self.max_num != 0)
        # default max_epochs/iters =0 if not epoch/iter_runner
        if not self.by_epoch:
            self.max_num = self.message_hub._runtime_info['max_iters']

        if self.by_epoch:
            cur_num = self.message_hub.get_info('epoch')
        else:
            cur_num = self.message_hub.get_info('iter')

        if (cur_num % self.fuse_count == 0):
            temperature = self._calc_temperature(cur_num, self.max_num)
            self.mutator.calc_information(temperature)

        batch_inputs, data_samples = self.data_preprocessor(data, True)
        with optim_wrapper.optim_context(self):
            losses = self(batch_inputs, data_samples, mode='loss')
        # parsed_losses, _ = self.module.parse_losses(losses)
        parsed_losses, _ = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)

        return losses

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE,
                       channel_cfgs: Union[str, Dict]) -> DCFFChannelMutator:
        """build mutator."""
        if isinstance(mutator, dict):
            assert 'channel_cfgs' not in mutator
            mutator['channel_cfgs'] = channel_cfgs
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, DCFFChannelMutator):
            raise TypeError('mutator should be a `dict` or '
                            '`DCFFChannelMutator` instance, but got '
                            f'{type(mutator)}')

        return mutator

    @property
    def _optim_wrapper_count_status_reinitialized(self) -> bool:
        return self.module._optim_wrapper_count_status_reinitialized

    @_optim_wrapper_count_status_reinitialized.setter
    def _optim_wrapper_count_status_reinitialized(self, val: bool) -> None:
        assert isinstance(val, bool)

        self.module._optim_wrapper_count_status_reinitialized = val
