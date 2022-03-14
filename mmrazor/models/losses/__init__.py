# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .weighted_soft_label_distillation import WSLD
from .rank_mimic import RankMimicLoss
from .pgfi import PredictionGuidedFeatureImitation

__all__ = [
    'PredictionGuidedFeatureImitation','ChannelWiseDivergence', 
    'KLDivergence', 'WSLD', 'RankMimicLoss']
