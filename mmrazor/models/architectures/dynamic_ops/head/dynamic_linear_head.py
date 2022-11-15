# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Optional, Tuple

import torch
from mmcls.models import ClsHead

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.mutables.mutable_channel import MutableChannelContainer
from mmrazor.models.mutables.mutable_channel.units import \
    OneShotMutableChannelUnit
from mmrazor.registry import MODELS
from ..bricks.dynamic_linear import DynamicLinear


class DynamicHead:

    @abstractmethod
    def connect_with_backbone(self,
                              backbone_output_mutable: BaseMutable) -> None:
        """Connect with Dynamic Backbone."""
        ...


@MODELS.register_module()
class DynamicLinearClsHead(ClsHead, DynamicHead):
    """Dynamic Linear classification head for Autoformer.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        init_cfg (Optional[dict], optional): Init config.
            Defaults to dict(type='Normal',
                        layer='DynamicLinear', std=0.01).
    """

    def __init__(self,
                 num_classes: int = 1000,
                 in_channels: int = 624,
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
        """Connect dynamic backbone."""

        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        MutableChannelContainer.register_mutable_channel_to_module(
            self.fc, backbone_output_mutable, False)
