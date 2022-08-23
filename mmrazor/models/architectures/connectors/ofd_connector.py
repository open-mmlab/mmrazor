# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class OFDConnector(BaseConnector):

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)
        self.margin: torch.Tensor = None

    def init_margin(self, margin):
        self.margin = margin
