# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .relational_kd import Angle_wise_RKD, Distance_wise_RKD
from .weighted_soft_label_distillation import WSLD

__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'Distance_wise_RKD',
    'Angle_wise_RKD', 'WSLD'
]
