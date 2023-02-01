# Copyright (c) OpenMMLab. All rights reserved.
from .distill_loss_weight_scheduler import (CosineAnnealingLossWeightScheduler,
                                            LinearLossWeightScheduler,
                                            LossWeightScheduler,
                                            LossWeightSchedulerManager,
                                            MultiStepLossWeightScheduler)

__all__ = [
    'CosineAnnealingLossWeightScheduler', 'LossWeightScheduler',
    'MultiStepLossWeightScheduler', 'LinearLossWeightScheduler',
    'LossWeightSchedulerManager'
]
