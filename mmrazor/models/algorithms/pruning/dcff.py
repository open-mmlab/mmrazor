# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutators import DCFFChannelMutator
from mmrazor.models.utils import reinitialize_optim_wrapper_count_status
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[DCFFChannelMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]
VALID_PATH_TYPE = Union[str, Path]
VALID_CHANNEL_CFG_PATH_TYPE = Union[VALID_PATH_TYPE, List[VALID_PATH_TYPE]]


@MODELS.register_module()
class DCFF(BaseAlgorithm):
    """DCFF Networks.

    Please refer to paper
    [Dynamic-coded Filter Fusion](https://arxiv.org/abs/2107.06916).

    Args:
        mutator (dict | :obj:`SlimmableChannelMutator`): The config of
            :class:`SlimmableChannelMutator` or built mutator.
            About the config of mutator, please refer to
            SlimmableChannelMutator
        architecture (dict | :obj:`BaseModel`): The config of
            :class:`BaseModel` or built model.
        deploy_index (int): index of subnet to be deployed.
        data_preprocessor (dict | :obj:`torch.nn.Module` | None): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        fuse_freq (int): frequency of filter fusion (epoch/iter runner).
        init_cfg (dict | None): The weight initialized config for
            :class:`BaseModule`. Default to None.
    """

    def __init__(self,
                 mutator: VALID_MUTATOR_TYPE,
                 architecture: Union[BaseModel, Dict],
                 deploy_index: int = -1,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 fuse_freq: int = 1,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        if isinstance(mutator, dict):
            self.mutator = MODELS.build(mutator)
        else:
            self.mutator = mutator
        self.mutator.prepare_from_supernet(self.architecture)
        self.num_subnet = len(self.mutator.subnets)
        self.fuse_freq = fuse_freq

        # must after `prepare_from_supernet`
        if deploy_index != -1:
            self._deploy(deploy_index)
        else:
            self.is_deployed = False

        self._optim_wrapper_count_status_reinitialized = False

    def _fix_archtecture(self):
        for module in self.architecture.modules():
            if isinstance(module, BaseMutable):
                if not module.is_fixed:
                    module.fix_chosen(None)

    def _deploy(self, index: int):
        self.mutator.set_choices(self.mutator.subnets[index])
        self.mutator.fix_channel_mutables()
        self._fix_archtecture()
        _dynamic_to_static(self.architecture)
        self.is_deployed = True

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

        if (cur_num % self.fuse_freq == 0):
            temperature = self._calc_temperature(cur_num, self.max_num)
            self.mutator.calc_information(temperature)

        batch_inputs, data_samples = self.data_preprocessor(data, True)
        # DCFF supports single subnet
        self.mutator.set_choices(self.mutator.subnets[0])
        with optim_wrapper.optim_context(self):
            losses = self(batch_inputs, data_samples, mode='loss')
        parsed_losses, _ = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)

        return losses
