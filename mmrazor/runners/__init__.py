# Copyright (c) OpenMMLab. All rights reserved.
from .darts_loop import DartsEpochBasedTrainLoop, DartsIterBasedTrainLoop
from .distill_val_loop import SingleTeacherDistillValLoop
from .evolution_search_loop import EvolutionSearchLoop
from .subnet_sampler_loop import GreedySamplerTrainLoop

__all__ = [
    'SingleTeacherDistillValLoop', 'DartsEpochBasedTrainLoop',
    'DartsIterBasedTrainLoop', 'EvolutionSearchLoop', 'GreedySamplerTrainLoop'
]
