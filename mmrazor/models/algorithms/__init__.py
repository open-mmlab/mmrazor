# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseAlgorithm
from .distill import (DAFLDataFreeDistillation, DataFreeDistillation,
                      FpnTeacherDistill, OverhaulFeatureDistillation,
                      SelfDistill, SingleTeacherDistill)
from .nas import DSNAS, DSNASDDP, SPOS, AutoSlim, AutoSlimDDP, Darts, DartsDDP
from .pruning import SlimmableNetwork, SlimmableNetworkDDP
from .pruning.ite_prune_algorithm import ItePruneAlgorithm
from .quantization import GeneralQuant

__all__ = [
    # base
    'BaseAlgorithm',
    # distill
    'DAFLDataFreeDistillation',
    'DataFreeDistillation',
    'FpnTeacherDistill',
    'OverhaulFeatureDistillation',
    'SelfDistill',
    'SingleTeacherDistill',
    # nas
    'DSNAS',
    'DSNASDDP',
    'SPOS',
    'AutoSlim',
    'AutoSlimDDP',
    'Darts',
    'DartsDDP',
    # pruning
    'SlimmableNetwork',
    'SlimmableNetworkDDP',
    'ItePruneAlgorithm',
    # quantization
    'GeneralQuant'
]
