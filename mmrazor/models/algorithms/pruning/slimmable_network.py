# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutators import SlimmableChannelMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[SlimmableChannelMutator, Dict]
VALID_PATH_TYPE = Union[str, Path]
VALID_CHANNEL_CFG_PATH_TYPE = Union[VALID_PATH_TYPE, List[VALID_PATH_TYPE]]


@MODELS.register_module()
class SlimmableNetwork(BaseAlgorithm):
    """Slimmable Neural Networks.

    Please refer to paper
    [Slimmable Neural Networks](https://arxiv.org/abs/1812.08928) for details.

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
        init_cfg (dict | None): The weight initialized config for
            :class:`BaseModule`. Default to None.
    """

    def __init__(self,
                 mutator: VALID_MUTATOR_TYPE,
                 architecture: Union[BaseModel, Dict],
                 deploy_index=-1,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        if isinstance(mutator, dict):
            self.mutator = MODELS.build(mutator)
        else:
            self.mutator = mutator
        self.mutator.prepare_from_supernet(self.architecture)
        self.num_subnet = len(self.mutator.subnets)

        # must after `prepare_from_supernet`
        if deploy_index != -1:
            self._deploy(deploy_index)
        else:
            self.is_deployed = False

        # HACK
        # reinitialize count status of `OptimWrapper` since
        # `optim_wrapper.update_params` will be called multiple times
        # in our slimmable train step.
        self._optim_wrapper_count_status_reinitialized = False

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Train step."""
        input_data = self.data_preprocessor(data, True)
        batch_inputs = input_data['inputs']
        data_samples = input_data['data_samples']
        train_kwargs = dict(
            batch_inputs=batch_inputs,
            data_samples=data_samples,
            optim_wrapper=optim_wrapper)
        if self.is_deployed:
            return self._fixed_train_step(**train_kwargs)
        else:
            return self._slimmable_train_step(**train_kwargs)

    def _slimmable_train_step(
        self,
        batch_inputs: torch.Tensor,
        data_samples: List[BaseDataElement],
        optim_wrapper: OptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        """Train step of Slimmable Network."""
        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=self.num_subnet)
            self._optim_wrapper_count_status_reinitialized = True
        total_losses = dict()

        for subnet_idx, subnet in enumerate(self.mutator.subnets):
            self.mutator.set_choices(subnet)
            with optim_wrapper.optim_context(self):
                losses = self(batch_inputs, data_samples, mode='loss')
            parsed_losses, _ = self.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)

            total_losses.update(add_prefix(losses, f'subnet_{subnet_idx}'))

        return total_losses

    def _fixed_train_step(
        self,
        batch_inputs: torch.Tensor,
        data_samples: List[BaseDataElement],
        optim_wrapper: OptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        """Train step of fixed network."""
        with optim_wrapper.optim_context(self):
            losses = self(batch_inputs, data_samples, mode='loss')
        parsed_losses, _ = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)

        return losses

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


@MODEL_WRAPPERS.register_module()
class SlimmableNetworkDDP(MMDistributedDataParallel):
    """DDP wrapper for Slimmable Neural Network."""

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Train step."""
        input_data = self.module.data_preprocessor(data, True)
        batch_inputs = input_data['inputs']
        data_samples = input_data['data_samples']
        train_kwargs = dict(
            batch_inputs=batch_inputs,
            data_samples=data_samples,
            optim_wrapper=optim_wrapper)
        if self.module.is_deployed:
            return self._fixed_train_step(**train_kwargs)
        else:
            return self._slimmable_train_step(**train_kwargs)

    def _slimmable_train_step(
        self,
        batch_inputs: torch.Tensor,
        data_samples: List[BaseDataElement],
        optim_wrapper: OptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        """Train step of Slimmable Network."""
        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=self.module.num_subnet)
            self._optim_wrapper_count_status_reinitialized = True
        total_losses = dict()

        for subnet_idx, subnet in enumerate(self.module.mutator.subnets):
            self.module.mutator.set_choices(subnet)
            with optim_wrapper.optim_context(self):
                losses = self(batch_inputs, data_samples, mode='loss')
            parsed_losses, _ = self.module.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)

            total_losses.update(add_prefix(losses, f'subnet_{subnet_idx}'))

        return total_losses

    def _fixed_train_step(
        self,
        batch_inputs: torch.Tensor,
        data_samples: List[BaseDataElement],
        optim_wrapper: OptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        """Train step of fixed network."""
        with optim_wrapper.optim_context(self):
            losses = self(batch_inputs, data_samples, mode='loss')
        parsed_losses, _ = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)

        return losses

    @property
    def _optim_wrapper_count_status_reinitialized(self) -> bool:
        return self.module._optim_wrapper_count_status_reinitialized

    @_optim_wrapper_count_status_reinitialized.setter
    def _optim_wrapper_count_status_reinitialized(self, val: bool) -> None:
        assert isinstance(val, bool)

        self.module._optim_wrapper_count_status_reinitialized = val
