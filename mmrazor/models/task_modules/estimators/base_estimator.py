# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

import torch.nn

from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BaseEstimator(metaclass=ABCMeta):
    """The base class of Estimator, used for estimating model infos.

    Args:
        input_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224).
        units (dict): A dict including required units. Default to dict().
        as_strings (bool): Output FLOPs and params counts in a string
            form. Default to False.
    """

    def __init__(self,
                 input_shape: Tuple = (1, 3, 224, 224),
                 units: Dict = dict(),
                 as_strings: bool = False):
        assert len(input_shape) in [
            3, 4, 5
        ], ('The length of input_shape must be in [3, 4, 5]. '
            f'Got `{len(input_shape)}`.')
        self.input_shape = input_shape
        self.units = units
        self.as_strings = as_strings

    @abstractmethod
    def estimate(self,
                 model: torch.nn.Module,
                 flops_params_cfg: dict = None,
                 latency_cfg: dict = None) -> Dict[str, Union[float, str]]:
        """Estimate the resources(flops/params/latency) of the given model.

        Args:
            model: The measured model.
            flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
                Default to None.
            latency_cfg (dict): Cfg for estimating latency. Default to None.

        Returns:
            Dict[str, Union[float, str]]): A dict that contains the resource
                results(FLOPs, params and latency).
        """
        pass
