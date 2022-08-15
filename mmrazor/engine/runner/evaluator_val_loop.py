# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, List, Union

from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop
from mmrazor.registry import ESTIMATOR, LOOPS
from mmrazor.structures import BaseEstimator


@LOOPS.register_module()
class EvaluatorLoop(ValLoop):
    """Loop for validation with BaseEstimator.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        estimator (Dict[str, Any]): Used for building a resource estimator.
            Default to be dict().
        resource_args (Dict[str, Any]): Resource information for estimator.
        NOTE: resource_args accept the following input items():
            input_shape (tuple): Input shape (including batchsize) used for
                model resources calculation.
            measure_inference (bool): whether to measure infer speed or not.
                Default to False.
            disabled_counters (list): One can limit which ops' spec would be
                calculated. Default to `None`.
            as_strings (bool): Output FLOPs and params counts in a string
                form. Default to True.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 estimator_cfg: Dict = dict(),
                 resource_args: Dict = dict(),
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.resource_args = resource_args
        if estimator_cfg:
            self.estimator: BaseEstimator = ESTIMATOR.build(estimator_cfg)
        else:
            self.estimator = None  # type: ignore

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.estimator:
            resource_results = self.estimate_resources(self.resource_args)
            metrics.update(resource_results)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    def estimate_resources(self, resource_args: Dict[str,
                                                     Any]) -> Dict[str, float]:
        """Estimate model resources: latency/flops/capacity."""
        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        # TODO confirm the state judgement.
        if model.is_supernet:
            model = self.export_subnet(model)

        resource_results = self.estimator.evaluate(
            model=model, resource_args=resource_args)

        return resource_results

    def export_subnet(self, model):
        """Export current best subnet."""
        # Avoid circular import
        from mmrazor.models.mutables.base_mutable import BaseMutable
        from mmrazor.structures import load_fix_subnet

        # delete non-leaf tensor to get deepcopy(model).
        # TODO solve the hard case.
        for module in model.architecture.modules():
            if isinstance(module, BaseMutable):
                if hasattr(module, 'arch_weights'):
                    delattr(module, 'arch_weights')

        copied_model = copy.deepcopy(model)
        fix_mutable = copied_model.search_subnet()
        load_fix_subnet(copied_model, fix_mutable)

        return copied_model
