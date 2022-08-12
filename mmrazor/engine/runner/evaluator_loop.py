# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Union

from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop
from mmrazor.registry import LOOPS


@LOOPS.register_module()
class EvaluatorLoop(ValLoop):
    """Loop for validation with NaiveEvaluator.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        if model.is_supernet:
            model = self.export_subnet(model)
        # compute metrics with resources(latency/flops/capacity) evaluated.
        metrics = self.evaluator.evaluate(
            len(self.dataloader.dataset),
            eval_resources=True,
            model=model,
            resource_args=dict(input_shape=self.evaluator.default_shape))
        del model
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    def export_subnet(self, model):
        """Export current best subnet."""
        # Avoid circular import
        from mmrazor.models.mutables.base_mutable import BaseMutable
        from mmrazor.structures import load_fix_subnet

        # delete non-leaf tensor to get deepcopy(model)
        for module in model.architecture.modules():
            if isinstance(module, BaseMutable):
                if hasattr(module, 'arch_weights'):
                    delattr(module, 'arch_weights')

        copied_model = copy.deepcopy(model)
        fix_mutable = copied_model.search_subnet()
        load_fix_subnet(copied_model, fix_mutable)
        return copied_model
