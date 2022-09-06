# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures import SwitchableBatchNorm2d


def switchable_bn_converter(module: _BatchNorm) -> SwitchableBatchNorm2d:
    """Convert a _BatchNorm module to a SwitchableBatchNorm2d.

    Args:
        module (:obj:`torch.nn.GroupNorm`): The original BatchNorm module.
    """
    switchable_bn = SwitchableBatchNorm2d.convert_from(module)

    return switchable_bn
