# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mmengine.model import BaseModule

from ..algorithms.base import LossResults


class BaseDistiller(BaseModule, ABC):
    """Base class for distiller.

    Args:
        calculate_student_loss (bool): Whether to calculate student loss
            (original task loss) to update student model. Defaults to True.
        init_cfg (dict, optional): Config for distiller. Default to None.
    """

    def __init__(self,
                 calculate_student_loss: bool = True,
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.calculate_student_loss = calculate_student_loss

    @abstractmethod
    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
