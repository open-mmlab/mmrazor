# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch.nn

from mmrazor.registry import TASK_UTILS
from .base_estimator import BaseEstimator
from .counters import get_model_complexity_info, repeat_measure_inference_speed


@TASK_UTILS.register_module()
class ResourceEstimator(BaseEstimator):
    """Estimator for calculating the resources consume.

    Args:
        input_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224).
        units (dict): Dict that contains converted FLOPs/params/latency units.
            Default to dict(flops='M', params='M', latency='ms').
        as_strings (bool): Output FLOPs/params/latency counts in a string
            form. Default to False.
        spec_modules (list): List of spec modules that needed to count.
            e.g., ['backbone', 'head'], ['backbone.layer1']. Default to [].
        disabled_counters (list): List of disabled spec op counters.
            It contains the op counter names in estimator.op_counters that
            are required to be disabled, e.g., ['BatchNorm2dCounter'].
            Defaults to [].
        measure_latency (bool): whether to measure inference speed or not.
            Default to False.
        latency_max_iter (Optional[int]): Max iteration num for the
            measurement. Default to 100.
        latency_num_warmup (Optional[int]): Iteration num for warm-up stage.
            Default to 5.
        latency_log_interval (Optional[int]): Interval num for logging the
            results. Default to 100.
        latency_repeat_num (Optional[int]): Num of times to repeat the
            measurement. Default to 1.

    Examples:
        >>> # direct calculate resource consume of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> estimator = ResourceEstimator()
        >>> estimator.estimate(
        ...     model=conv2d,
        ...     input_shape=(1, 3, 64, 64))
        {'flops': 3.444, 'params': 0.001, 'latency': 0.0}

        >>> # calculate resources of custom modules
        >>> class CustomModule(nn.Module):
        ...
        ...    def __init__(self) -> None:
        ...        super().__init__()
        ...
        ...    def forward(self, x):
        ...        return x
        ...
        >>> @TASK_UTILS.register_module()
        ... class CustomModuleCounter(BaseCounter):
        ...
        ...    @staticmethod
        ...    def add_count_hook(module, input, output):
        ...        module.__flops__ += 1000000
        ...        module.__params__ += 700000
        ...
        >>> model = CustomModule()
        >>> estimator.estimate(
        ...     model=model,
        ...     input_shape=(1, 3, 64, 64))
        {'flops': 1.0, 'params': 0.7, 'latency': 0.0}
        ...
        >>> # calculate resources of custom modules with disable_counters
        >>> estimator.estimate(
        ...     model=model,
        ...     input_shape=(1, 3, 64, 64),
        ...     disabled_counters=['CustomModuleCounter'])
        {'flops': 0.0, 'params': 0.0, 'latency': 0.0}

        >>> # calculate resources of mmrazor.models
        NOTE: check 'EstimateResourcesHook' in
              mmrazor.engine.hooks.estimate_resources_hook for details.
    """

    def __init__(
        self,
        input_shape: Tuple = (1, 3, 224, 224),
        units: Dict = dict(flops='M', params='M', latency='ms'),
        as_strings: bool = False,
        spec_modules: List[str] = [],
        disabled_counters: List[str] = [],
        measure_latency: bool = False,
        latency_max_iter: int = 100,
        latency_num_warmup: int = 5,
        latency_log_interval: int = 100,
        latency_repeat_num: int = 1,
    ):
        super().__init__(input_shape, units, as_strings)
        self.spec_modules = spec_modules
        self.disabled_counters = disabled_counters

        self.measure_latency = measure_latency
        self.latency_max_iter = latency_max_iter
        self.latency_num_warmup = latency_num_warmup
        self.latency_log_interval = latency_log_interval
        self.latency_repeat_num = latency_repeat_num

    def estimate(self, model: torch.nn.Module,
                 **kwargs) -> Dict[str, Union[float, str]]:
        """Estimate the resources(flops/params/latency) of the given model.

        Args:
            model: The measured model.

        Returns:
            Dict[str, str]): A dict that containing resource results(flops,
                params and latency).
        """
        latency_cfg = dict()
        resource_metrics = dict()
        for key in self.__init__.__code__.co_varnames[1:]:  # type: ignore
            kwargs.setdefault(key, getattr(self, key))
            if 'latency' in key:
                latency_cfg[key] = kwargs.pop(key)
        latency_cfg['unit'] = kwargs['units'].get('latency')
        latency_cfg['as_strings'] = kwargs['as_strings']
        latency_cfg['input_shape'] = kwargs['input_shape']

        model.eval()
        flops, params = get_model_complexity_info(model, **kwargs)

        if latency_cfg['measure_latency']:
            latency = repeat_measure_inference_speed(model, **latency_cfg)
        else:
            latency = '0.0 ms' if kwargs['as_strings'] else 0.0  # type: ignore

        resource_metrics.update({
            'flops': flops,
            'params': params,
            'latency': latency
        })
        return resource_metrics

    def estimate_separation_modules(self, model: torch.nn.Module,
                                    **kwargs) -> Dict[str, Union[float, str]]:
        """Estimate the resources(flops/params/latency) of the spec modules.

        Args:
            model: The measured model.

        Returns:
            Dict[str, float]): A dict that containing resource results(flops,
                params) of each modules in kwargs['spec_modules'].
        """
        for key in self.__init__.__code__.co_varnames[1:]:  # type: ignore
            kwargs.setdefault(key, getattr(self, key))
            # TODO: support speed estimation for separation modules.
            if 'latency' in key:
                kwargs.pop(key)

        assert len(kwargs['spec_modules']), (
            f'spec_modules can not be empty when calling '
            f'{self.__class__.__name__}.estimate_separation_modules().')
        kwargs['seperate_return'] = True

        model.eval()
        spec_modules_resources = get_model_complexity_info(model, **kwargs)
        return spec_modules_resources
