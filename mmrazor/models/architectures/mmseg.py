# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import MODELS
from .base import BaseArchitecture


@MODELS.register_module()
class MMSegArchitecture(BaseArchitecture):
    """Architecture based on MMSeg."""

    def __init__(self, **kwargs):
        super(MMSegArchitecture, self).__init__(**kwargs)
