# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from mmengine import fileio
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.mutators import SlimmableChannelMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS
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
        architecture (dict | :obj:`BaseModel`): The config of
            :class:`BaseModel` or built model.
        channel_cfg_paths (str | :obj:`Path` | list): Config of list of configs
            for channel of subnet(s) searched out. If there is only one
            channel_cfg, the supernet will be fixed.
        data_preprocessor (dict | :obj:`torch.nn.Module` | None): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        init_cfg (dict | None): The weight initialized config for
            :class:`BaseModule`. Default to None.
    """

    def __init__(self,
                 mutator: VALID_MUTATOR_TYPE,
                 architecture: Union[BaseModel, Dict],
                 channel_cfg_paths: VALID_CHANNEL_CFG_PATH_TYPE,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        if not isinstance(channel_cfg_paths, list):
            channel_cfg_paths = [channel_cfg_paths]
        self.num_subnet = len(channel_cfg_paths)

        channel_cfgs = self._load_and_merge_channel_cfgs(channel_cfg_paths)
        if isinstance(mutator, dict):
            assert mutator.get('channel_cfgs') is None, \
                '`channel_cfgs` should not be in channel config'
            mutator = copy.deepcopy(mutator)
            mutator['channel_cfgs'] = channel_cfgs

        self.mutator: SlimmableChannelMutator = self._build_mutator(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

        # must after `prepare_from_supernet`
        if len(channel_cfg_paths) == 1:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet
            load_fix_subnet(self.architecture, channel_cfg_paths[0])
            self.is_deployed = True
        else:
            self.is_deployed = False

        # HACK
        # reinitialize count status of `OptimWrapper` since
        # `optim_wrapper.update_params` will be called multiple times
        # in our slimmable train step.
        self._optim_wrapper_count_status_reinitialized = False

    def _load_and_merge_channel_cfgs(
            self, channel_cfg_paths: List[VALID_PATH_TYPE]) -> Dict:
        """Load and merge channel config."""
        channel_cfgs = list()
        for channel_cfg_path in channel_cfg_paths:
            channel_cfg = fileio.load(channel_cfg_path)
            channel_cfgs.append(channel_cfg)

        return self.merge_channel_cfgs(channel_cfgs)

    @staticmethod
    def merge_channel_cfgs(channel_cfgs: List[Dict]) -> Dict:
        """Merge several channel configs."""
        merged_channel_cfg = dict()
        num_subnet = len(channel_cfgs)

        for module_name in channel_cfgs[0].keys():
            channels_per_layer = [
                channel_cfgs[idx][module_name] for idx in range(num_subnet)
            ]
            merged_channels_per_layer = dict()
            for key in channels_per_layer[0].keys():
                merged_channels = [
                    channels_per_layer[idx][key] for idx in range(num_subnet)
                ]
                merged_channels_per_layer[key] = merged_channels
            merged_channel_cfg[module_name] = merged_channels_per_layer

        return merged_channel_cfg

    def _build_mutator(self,
                       mutator: VALID_MUTATOR_TYPE) -> SlimmableChannelMutator:
        """build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, SlimmableChannelMutator):
            raise TypeError('mutator should be a `dict` or '
                            '`SlimmableChannelMutator` instance, but got '
                            f'{type(mutator)}')

        return mutator

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Train step."""
        batch_inputs, data_samples = self.data_preprocessor(data, True)
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

        for subnet_idx in range(self.num_subnet):
            self.mutator.switch_choices(subnet_idx)
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
        batch_inputs, data_samples = self.module.data_preprocessor(data, True)
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

        for subnet_idx in range(self.module.num_subnet):
            self.module.mutator.switch_choices(subnet_idx)
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
