# Copyright (c) OpenMMLab. All rights reserved.
from .autoformer import Autoformer
from .autoslim import AutoSlim, AutoSlimDDP
from .darts import Darts, DartsDDP
from .dsnas import DSNAS, DSNASDDP
from .spos import SPOS

__all__ = [
    'SPOS', 'AutoSlim', 'AutoSlimDDP', 'Darts', 'DartsDDP', 'DSNAS',
    'DSNASDDP', 'Autoformer'
]
