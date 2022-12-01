# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseAlgorithm
from .distill import (DAFLDataFreeDistillation, DataFreeDistillation,
                      FpnTeacherDistill, OverhaulFeatureDistillation,
                      SelfDistill, SingleTeacherDistill)
from .nas import DSNAS, DSNASDDP, SPOS, AutoSlim, AutoSlimDDP, Darts, DartsDDP
from .pruning import SlimmableNetwork, SlimmableNetworkDDP
from .pruning.ite_prune_algorithm import ItePruneAlgorithm
from .quantization import MMArchitectureQuant, MMArchitectureQuantDDP

__all__ = [
    'SingleTeacherDistill', 'BaseAlgorithm', 'FpnTeacherDistill', 'SPOS',
    'SlimmableNetwork', 'SlimmableNetworkDDP', 'AutoSlim', 'AutoSlimDDP',
    'Darts', 'DartsDDP', 'SelfDistill', 'DataFreeDistillation',
    'DAFLDataFreeDistillation', 'OverhaulFeatureDistillation',
    'ItePruneAlgorithm', 'DSNAS', 'DSNASDDP', 'MMArchitectureQuant',
    'MMArchitectureQuantDDP'
]
