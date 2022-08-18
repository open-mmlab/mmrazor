# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import torch.nn

from mmrazor.registry import ESTIMATORS


@ESTIMATORS.register_module()
class BaseEstimator(metaclass=ABCMeta):
    """The base class of Estimator, used for estimating model infos.

    Args:
        default_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224)
        units (str): Resource units. Defaults to 'M'.
        disable_counters (List[str]): Disable spec op counters.
            Defaults to None.
    """

    def __init__(self,
                 default_shape: Tuple = (1, 3, 224, 224),
                 units: str = 'M',
                 disabled_counters: List[str] = None):
        assert len(default_shape) in [3, 4, 5], \
            f'Unsupported shape: {default_shape}'
        self.default_shape = default_shape
        self.units = units
        self.disabled_counters = disabled_counters

    @abstractmethod
    def estimate(
        self, model: torch.nn.Module, resource_args: Dict[str, Any] = dict()
    ) -> Dict[str, float]:
        """Estimate the resources(flops/params/latency) of the given model.

        Args:
            model: The measured model.
            resource_args (Dict[str, float]): resources information.

        Returns:
            Dict[str, float]): A dict that containing resource results(flops,
                params and latency).
        """
        pass
