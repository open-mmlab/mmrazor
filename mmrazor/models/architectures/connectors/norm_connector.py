# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from mmcv.cnn import build_norm_layer

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class NormConnector(BaseConnector):

    def __init__(self, in_channels, norm_cfg, init_cfg: Optional[Dict] = None):
        super(NormConnector, self).__init__(init_cfg)
        _, self.norm = build_norm_layer(norm_cfg, in_channels)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        return self.norm(feature)
