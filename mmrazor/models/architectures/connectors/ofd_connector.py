# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class OFDTeacherConnector(BaseConnector):
    """Connector designed for ``OverhaulFeatureDistillation``

    Args:
        init_cfg (Optional[Dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)
        self.margin: torch.Tensor = None

    def init_margin(self, margin: torch.Tensor) -> None:
        """Initializing margin, will be called by
        ``OverhaulFeatureDistillation``.

        Args:
            margin (torch.Tensor): margin
        """
        self.margin = margin

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """forward func for training."""
        assert self.margin is not None, (
            'margin must be initialized before training.')
        self.margin = self.margin.to(feature.device)
        feature = torch.max(feature.detach(), self.margin)
        return feature
