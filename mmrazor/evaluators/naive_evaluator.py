# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Tuple, Union

import torch.nn

from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.evaluator import Evaluator
from mmengine.evaluator.metric import BaseMetric
from mmrazor.registry import MMRAZOR_EVALUATOR
from .latency_measure import repeat_measure_inference_speed
from .op_spec_counters import get_model_complexity_info, params_units_convert


@MMRAZOR_EVALUATOR.register_module()
class NaiveEvaluator(Evaluator):
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
                 metrics: Union[dict, BaseMetric, Sequence],
                 default_shape: Tuple[int] = (1, 3, 224, 224),
                 units: str = 'M',
                 test_fn=None,
                 disabled_counters=None):
        super(NaiveEvaluator, self).__init__(metrics=metrics)
        assert len(default_shape) in [3, 4, 5], \
                    f'Unsupported shape: {default_shape}'
        self.default_shape = default_shape
        self.units = units
        self.test_fn = test_fn
        self.disabled_counters = disabled_counters

    def evaluate(
        self,
        size: int,
        eval_resources: bool = False,
        model: torch.nn.Module = None,
        resource_args: Dict[str, float] = dict()
    ) -> Dict[str, float]:
        metrics = {}
        # step1. test metrics.
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)

        # step2. test resources.
        if eval_resources:
            res_results = self.evaluate_resource(model, resource_args)
            metrics.update(**res_results)

        return metrics

    def evaluate_resource(self, model: torch.nn.Module,
                          resource_args: Dict[str, float]) -> Dict[str, float]:
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
            resource_args['disabled_counters'] = self.disabled_counters
            model.eval()
            flops, capacity = get_model_complexity_info(model, **resource_args)
            if measure_inference:
                latency = repeat_measure_inference_speed(
                    model, resource_args, max_iter=100, repeat_num=2)
            else:
                latency = '0.0 ms' if isinstance(flops, str) else 0
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
