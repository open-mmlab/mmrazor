# Copyright (c) OpenMMLab. All rights reserved.
from .ab_loss import ABLoss
from .at_loss import ATLoss
from .crd_loss import CRDLoss
from .cross_entropy_loss import CrossEntropyLoss
from .cwd import ChannelWiseDivergence
from .dafl_loss import ActivationLoss, InformationEntropyLoss, OnehotLikeLoss
from .decoupled_kd import DKDLoss
from .dist_loss import DISTLoss
from .factor_transfer_loss import FTLoss
from .fbkd_loss import FBKDLoss
from .kd_soft_ce_loss import KDSoftCELoss
from .kl_divergence import KLDivergence
from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .mgd_loss import MGDLoss
from .ofd_loss import OFDLoss
from .pkd_loss import PKDLoss
from .relational_kd import AngleWiseRKD, DistanceWiseRKD
from .weighted_soft_label_distillation import WSLD

__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'AngleWiseRKD', 'DistanceWiseRKD',
    'WSLD', 'L2Loss', 'ABLoss', 'DKDLoss', 'KDSoftCELoss', 'ActivationLoss',
    'OnehotLikeLoss', 'InformationEntropyLoss', 'FTLoss', 'ATLoss', 'OFDLoss',
    'L1Loss', 'FBKDLoss', 'CRDLoss', 'CrossEntropyLoss', 'PKDLoss', 'MGDLoss',
    'DISTLoss'
]
