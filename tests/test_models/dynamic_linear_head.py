# Copyright (c) OpenMMLab. All rights reserved.
# Copyrigforward_trainht (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ..bricks import DynamicLinear
from .base import DynamicHead

try:
    from mmcls.models import ClsHead
except ImportError:
    from mmrazor.utils import get_placeholder
    ClsHead = get_placeholder('mmcls')


@MODELS.register_module()
class DynamicLinearClsHead(ClsHead, DynamicHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='DynamicLinear', std=0.01),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = DynamicLinear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score

    def connect_with_backbone(self,
                              backbone_output_mutable: BaseMutable) -> None:
        self.fc.register_mutable_attr('in_features', backbone_output_mutable)
