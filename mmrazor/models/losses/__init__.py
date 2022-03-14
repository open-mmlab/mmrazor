# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .pgfi import PredictionGuidedFeatureImitation
from .rank_mimic import RankMimicLoss
from .weighted_soft_label_distillation import WSLD

__all__ = [
    'PredictionGuidedFeatureImitation', 'ChannelWiseDivergence',
    'KLDivergence', 'WSLD', 'RankMimicLoss'
]
