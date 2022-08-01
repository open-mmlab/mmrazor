# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures import SwitchableBatchNorm2d


def switchable_bn_converter(
        module: _BatchNorm, mutable_cfgs: Dict,
        candidate_choices: List[int]) -> SwitchableBatchNorm2d:
    """Convert a _BatchNorm module to a SwitchableBatchNorm2d.

    Args:
        module (:obj:`torch.nn.GroupNorm`): The original BatchNorm module.
        num_channels_cfg (Dict): Config related to `num_features`.
    """
    switchable_bn = SwitchableBatchNorm2d(
        mutable_cfgs=mutable_cfgs,
        candidate_choices=candidate_choices,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)

    return switchable_bn
