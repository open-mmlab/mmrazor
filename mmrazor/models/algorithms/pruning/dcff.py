# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
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
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
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
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        if isinstance(channel_cfgs, str):
            channel_cfgs = fileio.load(channel_cfgs)

        # channel_cfgs = self._load_and_merge_channel_cfgs(channel_cfg_paths)
        self.mutator = self._build_mutator(copy.copy(mutator), channel_cfgs)
        self.mutator.prepare_from_supernet(self.architecture)
        self.num_subnet = len(self.mutator.subnets)

        # print(channel_cfgs)
        print(self.architecture)

        """
        # must after `prepare_from_supernet`
        if len(channel_cfg_paths) == 1:
            self.mutator.set_choices(self.mutator.subnets[0])
            self.mutator.fix_channel_mutables()
            self._fix_archtecture()
            _dynamic_to_static(self.architecture)
            self.is_deployed = True
        else:
            self.is_deployed = False
        """

        self._optim_wrapper_count_status_reinitialized = False
        # print("arch:", self.architecture)

    def _load_and_merge_channel_cfgs(
            self, channel_cfg_paths: List[VALID_PATH_TYPE]) -> Dict:
        """Load and merge channel config."""
        channel_cfgs = list()
        for channel_cfg_path in channel_cfg_paths:
            channel_cfg = fileio.load(channel_cfg_path)
            channel_cfgs.append(channel_cfg)

        return self.merge_channel_cfgs(channel_cfgs)

    def _fix_archtecture(self):
        for module in self.architecture.modules():
            if isinstance(module, BaseMutable):
                if not module.is_fixed:
                    module.fix_chosen(None)

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

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """The iteration step during training.

        This method defines an iteration step during training.
        TODO(sensecore): Note that each downstream task may also overide
        `train_step`, which may cause confused. We except a architecture rule
        for different downstream task, to regularize behavior they realized.
        Such as forward propagation, back propagation and optimizer updating.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \

                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def _build_mutator(self,
                       mutator: VALID_MUTATOR_TYPE,
                       channel_cfgs: List) -> DCFFChannelMutator:
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