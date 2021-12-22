# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ARCHITECTURES
from .base import BaseArchitecture


@ARCHITECTURES.register_module()
class MMSegArchitecture(BaseArchitecture):
    """Architecture based on MMSeg."""

    def __init__(self, **kwargs):
        super(MMSegArchitecture, self).__init__(**kwargs)
