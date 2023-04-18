# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseFakeQuantize
from .lsq import (LearnableFakeQuantize, enable_param_learning,
                  enable_static_estimate, enable_val)
from .torch_fake_quants import register_torch_fake_quants

__all__ = [
    'BaseFakeQuantize', 'register_torch_fake_quants', 'LearnableFakeQuantize',
    'enable_val', 'enable_param_learning', 'enable_static_estimate'
]
