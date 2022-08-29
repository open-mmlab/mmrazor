# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from mmengine.model import BaseModule


class BaseConnector(BaseModule, metaclass=ABCMeta):
    """Base class of connectors.

    Connector is mainly used for distillation, it usually converts the channel
    number of input feature to align features of student and teacher.

    All subclasses should implement the following APIs:

    - ``forward_train()``

    Args:
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.forward_train(feature)

    @abstractmethod
    def forward_train(
        self, feature: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Abstract train computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        pass
