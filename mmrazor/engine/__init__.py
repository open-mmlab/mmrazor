# Copyright (c) OpenMMLab. All rights reserved.
from .hooks import DumpSubnetHook, EstimateResourcesHook
from .optimizers import SeparateOptimWrapperConstructor
from .runner import (AutoSlimValLoop, DartsEpochBasedTrainLoop,
                     DartsIterBasedTrainLoop, EvolutionSearchLoop,
                     GreedySamplerTrainLoop, SelfDistillValLoop,
                     SingleTeacherDistillValLoop, SlimmableValLoop,
                     QATEpochBasedLoop, PTQLoop)

__all__ = [
    'SeparateOptimWrapperConstructor', 'DumpSubnetHook',
    'SingleTeacherDistillValLoop', 'DartsEpochBasedTrainLoop',
    'DartsIterBasedTrainLoop', 'SlimmableValLoop', 'EvolutionSearchLoop',
<<<<<<< HEAD
    'GreedySamplerTrainLoop', 'AutoSlimValLoop', 'EstimateResourcesHook',
    'SelfDistillValLoop'
=======
    'GreedySamplerTrainLoop', 'AutoSlimValLoop', 'SelfDistillValLoop',
<<<<<<< HEAD
    'EstimateResourcesHook', 'QATEpochBasedLoop'
>>>>>>> 4bcbb1d (remove CPatcher in custome_tracer)
=======
    'EstimateResourcesHook', 'QATEpochBasedLoop', 'PTQLoop'
>>>>>>> 61453e2 (adaround experiment)
]
