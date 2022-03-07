# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead
from torch import nn


@HEADS.register_module()
class LinearNoBiasClsHead(LinearClsHead):
    """Set bias to False in ``LinearClsHead``."""

    def __init__(self, **kwargs):
        super(LinearNoBiasClsHead, self).__init__(**kwargs)
        self.fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
