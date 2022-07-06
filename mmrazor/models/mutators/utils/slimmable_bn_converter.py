# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures import SwitchableBatchNorm2d


def switchable_bn_converter(module: _BatchNorm, in_channels_cfg: Dict,
                            out_channels_cfg: Dict) -> SwitchableBatchNorm2d:
    """Convert a _BatchNorm module to a SwitchableBatchNorm2d.

    Args:
        module (:obj:`torch.nn.GroupNorm`): The original BatchNorm module.
        num_channels_cfg (Dict): Config related to `num_features`.
    """
    switchable_bn = SwitchableBatchNorm2d(
        num_features_cfg=in_channels_cfg,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats)
    return switchable_bn
