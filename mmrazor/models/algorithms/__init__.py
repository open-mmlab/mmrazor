# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseAlgorithm
from .distill import (DAFLDataFreeDistillation, DataFreeDistillation,
                      FpnTeacherDistill, OverhaulFeatureDistillation,
                      SelfDistill, SingleTeacherDistill)
from .nas import SPOS, AutoSlim, AutoSlimDDP, Darts, DartsDDP, Dsnas, DsnasDDP
from .pruning import SlimmableNetwork, SlimmableNetworkDDP
<<<<<<< HEAD
from .pruning.ite_prune_algorithm import ItePruneAlgorithm
from .quantization import PTQ

__all__ = [
    'SingleTeacherDistill',
    'BaseAlgorithm',
    'FpnTeacherDistill',
    'SPOS',
    'SlimmableNetwork',
    'SlimmableNetworkDDP',
    'AutoSlim',
    'AutoSlimDDP',
    'Darts',
    'DartsDDP',
    'SelfDistill',
    'DataFreeDistillation',
    'DAFLDataFreeDistillation',
    'OverhaulFeatureDistillation',
    'ItePruneAlgorithm',
    'Dsnas',
    'DsnasDDP',
    'PTQ'
=======
from .quantization import QAT

__all__ = [
    'SingleTeacherDistill', 'BaseAlgorithm', 'FpnTeacherDistill', 'SPOS',
    'SlimmableNetwork', 'SlimmableNetworkDDP', 'AutoSlim', 'AutoSlimDDP',
    'Darts', 'DartsDDP', 'SelfDistill', 'DataFreeDistillation',
    'DAFLDataFreeDistillation', 'OverhaulFeatureDistillation',
    'QAT'
>>>>>>> 172d3fa (init version)
]
