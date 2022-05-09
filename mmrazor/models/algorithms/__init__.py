# Copyright (c) OpenMMLab. All rights reserved.
from .autoslim import AutoSlim
from .darts import Darts
from .detnas import DetNAS
from .general_distill import GeneralDistill
from .spos import SPOS

__all__ = ['AutoSlim', 'Darts', 'SPOS', 'DetNAS', 'GeneralDistill']
