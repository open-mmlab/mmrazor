# Copyright (c) OpenMMLab. All rights reserved.
from .ab_loss import ABLoss
from .crd_loss import CRDLoss
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .l2_loss import L2Loss
from .relational_kd import AngleWiseRKD, DistanceWiseRKD
from .weighted_soft_label_distillation import WSLD

__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'AngleWiseRKD', 'DistanceWiseRKD',
    'WSLD', 'L2Loss', 'ABLoss', 'CRDLoss'
]
