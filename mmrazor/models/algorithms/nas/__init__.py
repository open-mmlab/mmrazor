# Copyright (c) OpenMMLab. All rights reserved.
from .autoslim import AutoSlim, AutoSlimDDP
from .darts import Darts, DartsDDP
from .dsnas import Dsnas, DsnasDDP
from .spos import SPOS

__all__ = [
    'SPOS', 'AutoSlim', 'AutoSlimDDP', 'Darts', 'DartsDDP', 'Dsnas', 'DsnasDDP'
]
