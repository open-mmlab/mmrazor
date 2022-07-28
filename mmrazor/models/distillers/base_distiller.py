# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mmengine.model import BaseModule

from ..algorithms.base import LossResults


class BaseDistiller(BaseModule, ABC):
    """Base class for distiller.

    Args:
        init_cfg (dict, optional): Config for distiller. Default to None.
    """

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

    @abstractmethod
    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
