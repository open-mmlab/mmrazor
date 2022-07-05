# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables import MutableManageMixIn
from mmrazor.registry import MODELS
from .default_dynamic_ops import build_dynamic_conv2d, build_dynamic_linear


class SwitchableBatchNorm2d(nn.Module, MutableManageMixIn):
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
                 module_name: str,
                 num_features_cfg: Dict,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(SwitchableBatchNorm2d, self).__init__()

        num_features_cfg = copy.deepcopy(num_features_cfg)
        candidate_choices = num_features_cfg.pop('candidate_choices')
        num_features_cfg.update(
            dict(
                name=module_name,
                num_channels=max(candidate_choices),
                mask_type='out_mask'))

        bns = [
            nn.BatchNorm2d(num_features, eps, momentum, affine,
                           track_running_stats)
            for num_features in candidate_choices
        ]
        self.bns = nn.ModuleList(bns)

        self.mutable_num_features = MODELS.build(num_features_cfg)

    @property
    def mutable_out(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    def forward(self, input):
        """Forward computation according to the current switch of the slimmable
        networks."""
        idx = self.mutable_num_features.current_choice
        return self.bns[idx](input)


def build_switchable_bn(module: _BatchNorm, module_name: str,
                        num_features_cfg: Dict) -> SwitchableBatchNorm2d:
    """Build SwitchableBatchNorm2d.

    Args:
        module (:obj:`torch.nn.GroupNorm`): The original BatchNorm module.
        module_name (str): Name of this `SwitchableBatchNorm2d`.
        num_channels_cfg (Dict): Config related to `num_features`.
    """
    switchable_bn = SwitchableBatchNorm2d(
        module_name=module_name,
        num_features_cfg=num_features_cfg,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)
    return switchable_bn


SLIMMABLE_DYNAMIC_LAYER: Dict[Callable, Callable] = {
    nn.Conv2d: build_dynamic_conv2d,
    nn.Linear: build_dynamic_linear,
    nn.BatchNorm2d: build_switchable_bn
}
