# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS


@LOOPS.register_module()
class SlimmableValLoop(ValLoop):
    """Knowledge Distill loop for validation. It is not only validate student,
    but also validate teacher with the same dataloader.

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

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        # just for convenience
        self._model = model

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')

        all_metrics = dict()
        for subnet_idx, subnet in enumerate(self._model.mutator.subnets):
            self.runner.call_hook('before_val_epoch')
            self.runner.model.eval()
            self._model.mutator.set_choices(subnet)
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute student metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            all_metrics.update(add_prefix(metrics, f'subnet_{subnet_idx}'))

        self.runner.call_hook('after_val_epoch', metrics=all_metrics)

        self.runner.call_hook('after_val')
