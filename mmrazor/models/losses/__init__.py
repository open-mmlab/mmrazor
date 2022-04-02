# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .fgd import FGDLoss
from .kl_divergence import KLDivergence
from .weighted_soft_label_distillation import WSLD

__all__ = ['ChannelWiseDivergence', 'KLDivergence', 'WSLD', 'FGDLoss']
