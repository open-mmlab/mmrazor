# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .weighted_soft_label_distillation import WSLD
from .fgd import FGDLoss

__all__ = ['ChannelWiseDivergence', 'KLDivergence', 'WSLD', 'FGDLoss']
