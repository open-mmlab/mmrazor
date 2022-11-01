# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn

from mmrazor.registry import MODELS

try:
    from mmcls.models import VisionTransformerClsHead
except ImportError:
    from mmrazor.utils import get_placeholder
    VisionTransformerClsHead = get_placeholder('mmcls')


@MODELS.register_module()
class DeiTClsHead(VisionTransformerClsHead):
    """Distilled Vision Transformer classifier head.

    Comparing with the :class:`DeiTClsHead` in mmcls, this head support to
    train the distilled version DeiT.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int, optional): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to ``dict(type='Tanh')``.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """

    def _init_layers(self):
        """"Init extra hidden linear layer to handle dist token if exists."""
        super(DeiTClsHead, self)._init_layers()
        if self.hidden_dim is None:
            head_dist = nn.Linear(self.in_channels, self.num_classes)
        else:
            head_dist = nn.Linear(self.hidden_dim, self.num_classes)
        self.layers.add_module('head_dist', head_dist)

    def pre_logits(
            self, feats: Tuple[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage. In ``DeiTClsHead``, we obtain the
        feature of the last stage and forward in hidden layer if exists.
        """
        _, cls_token, dist_token = feats[-1]
        if self.hidden_dim is None:
            return cls_token, dist_token
        else:
            cls_token = self.layers.act(self.layers.pre_logits(cls_token))
            dist_token = self.layers.act(self.layers.pre_logits(dist_token))
            return cls_token, dist_token

    def forward(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The forward process."""
        cls_token, dist_token = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.layers.head(cls_token)
        # Forward so that the corresponding recorder can record the output
        # of the distillation token
        _ = self.layers.head_dist(dist_token)
        return cls_score
