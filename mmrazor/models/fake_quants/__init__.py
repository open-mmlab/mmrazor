# Copyright (c) OpenMMLab. All rights reserved.
from .adaround import AdaRoundFakeQuantize
from .base import FakeQuantize
from .lsq import LearnableFakeQuantize
from .qdrop import QDropFakeQuantize

__all__ = [
    'FakeQuantize', 'AdaRoundFakeQuantize', 'QDropFakeQuantize',
    'LearnableFakeQuantize'
]
