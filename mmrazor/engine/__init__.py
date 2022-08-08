# Copyright (c) OpenMMLab. All rights reserved.
from .hooks import DumpSubnetHook, EstimateResourcesHook
from .optimizers import SeparateOptimWrapperConstructor
from .runner import (AutoSlimValLoop, DartsEpochBasedTrainLoop,
                     DartsIterBasedTrainLoop, EvolutionSearchLoop,
                     GreedySamplerTrainLoop, SelfDistillValLoop,
                     SingleTeacherDistillValLoop, SlimmableValLoop,
                     QATEpochBasedLoop)

__all__ = [
    'SeparateOptimWrapperConstructor', 'DumpSubnetHook',
    'SingleTeacherDistillValLoop', 'DartsEpochBasedTrainLoop',
    'DartsIterBasedTrainLoop', 'SlimmableValLoop', 'EvolutionSearchLoop',
<<<<<<< HEAD
    'GreedySamplerTrainLoop', 'AutoSlimValLoop', 'EstimateResourcesHook',
    'SelfDistillValLoop'
=======
    'GreedySamplerTrainLoop', 'AutoSlimValLoop', 'SelfDistillValLoop',
    'EstimateResourcesHook', 'QATEpochBasedLoop'
>>>>>>> 4bcbb1d (remove CPatcher in custome_tracer)
]
