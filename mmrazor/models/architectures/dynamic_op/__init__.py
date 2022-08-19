# Copyright (c) OpenMMLab. All rights reserved.
from .base import DynamicOP
from .default_dynamic_ops import (DynamicBatchNorm, DynamicConv2d,
                                  DynamicGroupNorm, DynamicInstanceNorm,
                                  DynamicLinear)
from .slimmable_dynamic_ops import SwitchableBatchNorm2d

__all__ = [
    'DynamicConv2d', 'DynamicLinear', 'DynamicBatchNorm',
    'DynamicInstanceNorm', 'DynamicGroupNorm', 'SwitchableBatchNorm2d',
    'DynamicOP'
]
