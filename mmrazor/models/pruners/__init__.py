# Copyright (c) OpenMMLab. All rights reserved.
from .bcnet_pruning import BCNetPruner
from .ratio_pruning import RatioPruner
from .structure_pruning import StructurePruner
from .utils import *  # noqa: F401,F403

__all__ = ['BCNetPruner', 'RatioPruner', 'StructurePruner']
