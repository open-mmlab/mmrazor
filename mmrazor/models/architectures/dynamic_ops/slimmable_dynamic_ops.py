# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import torch.nn as nn

from mmrazor.models.mutables.mutable_channel import MutableChannel
from mmrazor.registry import MODELS
from .base import DynamicOP


class SwitchableBatchNorm2d(nn.Module, DynamicOP):
    """Employs independent batch normalization for different switches in a
    slimmable network.

    To train slimmable networks, ``SwitchableBatchNorm2d`` privatizes all
    batch normalization layers for each switch in a slimmable network.
    Compared with the naive training approach, it solves the problem of feature
    aggregation inconsistency between different switches by independently
    normalizing the feature mean and variance during testing.

    Args:
        module_name (str): Name of this `SwitchableBatchNorm2d`.
        num_features_cfg (Dict): Config related to `num_features`.
        eps (float): A value added to the denominator for numerical stability.
            Same as that in :obj:`torch.nn._BatchNorm`. Default: 1e-5
        momentum (float): The value used for the running_mean and running_var
            computation. Can be set to None for cumulative moving average
            (i.e. simple average). Same as that in :obj:`torch.nn._BatchNorm`.
            Default: 0.1
        affine (bool): A boolean value that when set to True, this module has
            learnable affine parameters. Same as that in
            :obj:`torch.nn._BatchNorm`. Default: True
        track_running_stats (bool): A boolean value that when set to True, this
            module tracks the running mean and variance, and when set to False,
            this module does not track such statistics, and initializes
            statistics buffers running_mean and running_var as None. When these
            buffers are None, this module always uses batch statistics. in both
            training and eval modes. Same as that in
            :obj:`torch.nn._BatchNorm`. Default: True
    """

    def __init__(self,
                 num_features_cfg: Dict,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(SwitchableBatchNorm2d, self).__init__()

        num_features_cfg = copy.deepcopy(num_features_cfg)
        candidate_choices = num_features_cfg.pop('candidate_choices')
        num_features_cfg.update(dict(num_channels=max(candidate_choices)))

        bns = [
            nn.BatchNorm2d(num_features, eps, momentum, affine,
                           track_running_stats)
            for num_features in candidate_choices
        ]
        self.bns = nn.ModuleList(bns)

        self.mutable_num_features = MODELS.build(num_features_cfg)

    @property
    def mutable_in(self) -> MutableChannel:
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self) -> MutableChannel:
        """Mutable `num_features`."""
        return self.mutable_num_features

    def forward(self, input):
        """Forward computation according to the current switch of the slimmable
        networks."""
        idx = self.mutable_num_features.current_choice
        return self.bns[idx](input)

    def to_static_op(self) -> nn.Module:
        bn_idx = self.mutable_num_features.current_choice

        return self.bns[bn_idx]
