# Copyright (c) OpenMMLab. All rights reserved.

from ..builder import ARCHITECTURES
from .base import BaseArchitecture


@ARCHITECTURES.register_module()
class MMDetArchitecture(BaseArchitecture):
    """Architecture based on MMDet."""

    def __init__(self, **kwargs):
        super(MMDetArchitecture, self).__init__(**kwargs)

    def cal_pseudo_loss(self, pseudo_img):
        """Used for executing ``forward`` with pseudo_img."""
        out = 0.
        for levels in pseudo_img:
            out += sum([level.sum() for level in levels])

        return out
