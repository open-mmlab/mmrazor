# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class BaseConnector(nn.Module, metaclass=ABCMeta):
    """Base class of connectors.

    Connector is mainly used for distill, it usually converts the channel
    number of input feature to align features of student and teacher.

    All subclasses should implement the following APIs:

    - ``forward_train()``
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature: torch.Tensor) -> None:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.forward_train(feature)

    @abstractmethod
    def forward_train(self, feature) -> torch.Tensor:
        """Abstract train computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        pass
