# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet
except (ImportError, ModuleNotFoundError):
    mmdet = None

try:
    import mmseg
except (ImportError, ModuleNotFoundError):
    mmseg = None

from mmcls.models import *  # noqa: F401,F403

from .algorithms import *  # noqa: F401,F403
from .architectures import *  # noqa: F401,F403
from .builder import (ALGORITHMS, ARCHITECTURES, DISTILLERS, LOSSES, MUTABLES,
                      MUTATORS, OPS, build_algorithm, build_architecture,
                      build_distiller, build_loss, build_mutable,
                      build_mutator, build_op)
from .distillers import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .mutables import *  # noqa: F401,F403
from .mutators import *  # noqa: F401,F403
from .ops import *  # noqa: F401,F403
from .pruners import *  # noqa: F401,F403

if mmdet is not None:
    from mmdet.models import *  # noqa: F401,F403
if mmseg is not None:
    from mmseg.models import *  # noqa: F401,F403

__all__ = [
    'ALGORITHMS', 'MUTABLES', 'ARCHITECTURES', 'DISTILLERS', 'MUTATORS',
    'LOSSES', 'OPS', 'build_architecture', 'build_mutable', 'build_op',
    'build_mutator', 'build_algorithm', 'build_distiller', 'build_loss'
]
