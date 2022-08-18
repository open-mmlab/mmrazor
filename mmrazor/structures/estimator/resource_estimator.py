# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch.nn
from mmengine.dist import broadcast_object_list, is_main_process

from mmrazor.registry import ESTIMATORS
from .base_estimator import BaseEstimator
from .flops_params_counter import (get_model_complexity_info,
                                   params_units_convert)
from .latency import repeat_measure_inference_speed


@ESTIMATORS.register_module()
class ResourceEstimator(BaseEstimator):
    """Estimator for calculating the resources consume.

    Args:
        default_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224)
        units (str): Resource units. Defaults to 'M'.
        disable_counters (list): Disable spec op counters.
            Defaults to None.
        NOTE: disable_counters contains the op counter class names
              in estimator.op_spec_counters that require to be disabled,
              such as 'ConvCounter', 'BatchNorm2dCounter', ...

    Examples:
        >>> # direct calculate resource consume of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> estimator = ResourceEstimator()
        >>> estimator.estimate(
        ...     model=conv2d,
        ...     resource_args=dict(input_shape=(1, 3, 64, 64)))
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
        >>> @OP_SPEC_COUNTERS.register_module()
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
        ...     resource_args=dict(input_shape=(1, 3, 64, 64)))
        {'flops': 1.0, 'params': 0.7, 'latency': 0.0}
        ...
        >>> # calculate resources of custom modules with disable_counters
        >>> estimator.estimate(
        ...     model=model,
        ...     resource_args=dict(
        ...         input_shape=(1, 3, 64, 64),
        ...         disabled_counters=['CustomModuleCounter']))
        {'flops': 0.0, 'params': 0.0, 'latency': 0.0}

        >>> # calculate resources of mmrazor.models
        NOTE: check 'ResourceEvaluatorLoop' in
              engine.runner.resource_evaluator_val_loop for more details.
    """

    def __init__(self,
                 default_shape: Tuple = (1, 3, 224, 224),
                 units: str = 'M',
                 disabled_counters: List[str] = None):
        super().__init__(default_shape, units, disabled_counters)

    def estimate(
        self, model: torch.nn.Module, resource_args: Dict[str, Any] = dict()
    ) -> Dict[str, Any]:
        """Estimate the resources(flops/params/latency) of the given model.

        NOTE: resource_args accept the following input items():
            input_shape (tuple): Input shape (including batchsize) used for
                model resources calculation.
            measure_inference (bool): whether to measure infer speed or not.
                Default to False.
            disabled_counters (list): One can limit which ops' spec would be
                calculated. Default to `None`.
            as_strings (bool): Output FLOPs and params counts in a string
                form. Default to False.
            add_resource_attr (bool): Whether to measure a model with adding
                `__flops__` and `__params__` attributes on each module of the
                model. Default to False.
        """
        results = dict()
        if is_main_process():
            measure_inference = resource_args.pop('measure_inference', False)
            if 'input_shape' not in resource_args.keys():
                resource_args['input_shape'] = self.default_shape
            if 'disabled_counters' not in resource_args.keys():
                resource_args['disabled_counters'] = self.disabled_counters
            model.eval()
            flops, params = get_model_complexity_info(model, **resource_args)
            if measure_inference:
                latency = repeat_measure_inference_speed(
                    model, resource_args, max_iter=100, repeat_num=2)
            else:
                latency = 0.0
            as_strings = resource_args.get('as_strings', False)
            if as_strings and self.units is not None:
                raise ValueError('Set units to None, when as_trings=True.')
            if self.units is not None:
                flops = params_units_convert(flops, self.units)
                params = params_units_convert(params, self.units)
            results.update({
                'flops': flops,
                'params': params,
                'latency': latency
            })
        broadcast_object_list([results])

        return results
