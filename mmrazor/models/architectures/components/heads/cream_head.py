# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional, Tuple

from mmcls.models.heads import LinearClsHead
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class CreamClsHead(LinearClsHead):
    """Linear classifier head for cream.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_features (int): Number of features in the conv2d.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_features: int = 1280,
                 act_cfg: Dict = dict(type='ReLU6'),
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=act_cfg)

        self.add_module('conv2', layer)

        self.fc = nn.Linear(num_features, self.num_classes)

    # def pre_logits(self, feats: Tuple[Tensor]) -> Tensor:
    #     """The process before the final classification head.

    #     The input ``feats`` is a tuple of tensor, and each tensor is the
    #     feature of a backbone stage. In ``LinearClsHead``, we just obtain the
    #     feature of the last stage.
    #     """
    #     # The LinearClsHead doesn't have other module, just return after
    #     # unpacking.
    #     return feats[-1]

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """The forward process."""
        logits = self.pre_logits(feats)
        logits = logits.unsqueeze(-1).unsqueeze(-1)
        logits = self.conv2(logits)
        logits = logits.flatten(1)
        return self.fc(logits)
