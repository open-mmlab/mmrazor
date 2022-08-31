# Copyright (c) OpenMMLab. All rights reserved.
from .autoformer import Autoformer
from .autoslim import AutoSlim, AutoSlimDDP
from .bignas import BigNAS, BigNASDDP
from .darts import Darts, DartsDDP
from .spos import SPOS

__all__ = [
    'SPOS', 'AutoSlim', 'AutoSlimDDP', 'BigNAS', 'BigNASDDP', 'Darts',
    'DartsDDP', 'Autoformer'
]
