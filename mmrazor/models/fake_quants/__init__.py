# Copyright (c) OpenMMLab. All rights reserved.
from .adaround import AdaRoundFakeQuantize
from .base import FakeQuantize
from .lsq import LearnableFakeQuantize
from .qdrop import QDropFakeQuantize
from .torch_fake_quants import register_torch_fake_quants

__all__ = [
    'FakeQuantize', 'AdaRoundFakeQuantize', 'QDropFakeQuantize',
    'LearnableFakeQuantize', 'register_torch_fake_quants'
]
