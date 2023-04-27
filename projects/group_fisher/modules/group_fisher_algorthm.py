# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.model import BaseModel, MMDistributedDataParallel

from mmrazor.models.algorithms.base import BaseAlgorithm
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ...cores.utils import RuntimeInfo  # type: ignore
from .group_fisher_channel_mutator import GroupFisherChannelMutator


@MODELS.register_module()
class GroupFisherAlgorithm(BaseAlgorithm):
    """`Group Fisher Pruning for Practical Network Compression`.
    https://arxiv.org/pdf/2108.00708.pdf.

    Args:
        architecture (Union[BaseModel, Dict]): The model to be pruned.
        mutator (Union[Dict, ChannelMutator], optional): The config
            of a mutator. Defaults to dict( type='GroupFisherChannelMutator',
            channel_unit_cfg=dict( type='GroupFisherChannelUnit')).
        interval (int): The interval of  pruning two channels. Defaults to 10.
        data_preprocessor (Optional[Union[Dict, nn.Module]], optional):
            Defaults to None.
        init_cfg (Optional[Dict], optional): init config for the model.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Union[Dict, GroupFisherChannelMutator] = dict(
                     type='GroupFisherChannelMutator',
                     channel_unit_cfg=dict(type='GroupFisherChannelUnit')),
                 interval: int = 10,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)

        self.interval = interval

        # using sync bn or normal bn
        if dist.is_initialized():
            print_log('Convert Bn to SyncBn.')
            self.architecture = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.architecture)
        else:
            from mmengine.model import revert_sync_batchnorm
            self.architecture = revert_sync_batchnorm(self.architecture)

        # mutator
        self.mutator: GroupFisherChannelMutator = MODELS.build(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:
        algorithm = self
        algorithm.mutator.start_record_info()
        res = super().train_step(data, optim_wrapper)
        algorithm.mutator.end_record_info()

        algorithm.mutator.update_imp()
        algorithm.mutator.reset_recorded_info()

        if RuntimeInfo.iter() % algorithm.interval == 0:
            algorithm.mutator.try_prune()
            algorithm.mutator.reset_imp()

        return res


@MODEL_WRAPPERS.register_module()
class GroupFisherDDP(MMDistributedDataParallel):
    """Train step for group fisher."""

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:
        algorithm = self.module
        algorithm.mutator.start_record_info()
        res = super().train_step(data, optim_wrapper)
        algorithm.mutator.end_record_info()

        algorithm.mutator.update_imp()
        algorithm.mutator.reset_recorded_info()

        if RuntimeInfo.iter() % algorithm.interval == 0:
            algorithm.mutator.try_prune()
            algorithm.mutator.reset_imp()

        return res
