# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import torch.nn

from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BaseEstimator(metaclass=ABCMeta):
    """The base class of Estimator, used for estimating model infos.

    Args:
        default_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224).
        units (str): Resource units. Defaults to 'M'.
        disabled_counters (list): List of disabled spec op counters.
            Defaults to None.
        as_strings (bool): Output FLOPs and params counts in a string
            form. Default to False.
        measure_inference (bool): whether to measure infer speed or not.
            Default to False.
    """

    def __init__(self,
                 default_shape: Tuple = (1, 3, 224, 224),
                 units: str = 'M',
                 disabled_counters: List[str] = None,
                 as_strings: bool = False,
                 measure_inference: bool = False):
        assert len(default_shape) in [3, 4, 5], \
            f'Unsupported shape: {default_shape}'
        self.default_shape = default_shape
        self.units = units
        self.disabled_counters = disabled_counters
        self.as_strings = as_strings
        self.measure_inference = measure_inference

    @abstractmethod
    def estimate(
        self, model: torch.nn.Module, resource_args: Dict[str, Any] = dict()
    ) -> Dict[str, float]:
        """Estimate the resources(flops/params/latency) of the given model.

        Args:
            model: The measured model.
            resource_args (Dict[str, float]): resources information.
            NOTE: resource_args have the same items() as the init cfgs.

        Returns:
            Dict[str, float]): A dict that containing resource results(flops,
                params and latency).
        """
        pass
