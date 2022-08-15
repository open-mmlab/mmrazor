# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Any, Dict, Tuple

import torch.nn

from mmengine.dist import broadcast_object_list, is_main_process
from mmrazor.registry import ESTIMATOR
from .flops_params_counter import (get_model_complexity_info,
                                   params_units_convert)
from .latency import repeat_measure_inference_speed
from .op_spec_counters import BaseCounter


@ESTIMATOR.register_module()
class BaseEstimator(metaclass=ABCMeta):
    """Evaluator for calculating the accuracy and resources consume. Accuracy
    calculation is optional.

    Args:
        default_shape (tuple): Input data's default shape, for calculating
            resources consume.
        units (str): Resource units. Defaults to 'M'.
        test_fn (callable, optional): Test function or callable class, for
            calculating accuracy. It should return a dict containing the
            accuracy indicator. Defaults to None.
    """

    def __init__(self,
                 default_shape: Tuple = (1, 3, 224, 224),
                 units: str = 'M',
                 disabled_counters: BaseCounter = None):
        assert len(default_shape) in [3, 4, 5], \
            f'Unsupported shape: {default_shape}'
        self.default_shape = default_shape
        self.units = units
        self.disabled_counters = disabled_counters

    def evaluate(
        self, model: torch.nn.Module, resource_args: Dict[str, Any] = dict()
    ) -> Dict[str, float]:
        """Evaluate the resources(latency/flops/capacity) of the given model.

        Args:
            model: The measured model.
            resource_args (Dict[str, float]): resources information.

        Returns:
            Dict[str, float]): A dict that containing resource results(flops,
                capacity and latency) and accuracy results.
        """
        results = dict()
        if is_main_process():
            measure_inference = resource_args.pop('measure_inference', False)
            if 'input_shape' not in resource_args.keys():
                resource_args['input_shape'] = self.default_shape
            resource_args['disabled_counters'] = self.disabled_counters
            model.eval()
            flops, capacity = get_model_complexity_info(model, **resource_args)
            if measure_inference:
                latency = repeat_measure_inference_speed(
                    model, resource_args, max_iter=100, repeat_num=2)
            else:
                latency = '0.0 ms'
            as_strings = resource_args.get('as_strings', False)
            if as_strings and self.units is not None:
                raise ValueError('Set units to None, when as_trings=True.')
            if self.units is not None:
                flops = params_units_convert(flops, self.units)
                capacity = params_units_convert(capacity, self.units)
            results.update({
                'flops': flops,
                'capacity': capacity,
                'latency': latency
            })
        broadcast_object_list([results])

        return results
