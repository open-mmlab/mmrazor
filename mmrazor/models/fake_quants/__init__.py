# Copyright (c) OpenMMLab. All rights reserved.
from .adaround import AdaRoundFakeQuantize
from .base import FakeQuantize
from .qdrop import QDropFakeQuantize

__all__ = ['FakeQuantize', 'AdaRoundFakeQuantize', 'QDropFakeQuantize']
