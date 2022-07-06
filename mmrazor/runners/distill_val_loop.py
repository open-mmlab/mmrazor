# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS


@LOOPS.register_module()
class SingleTeacherDistillValLoop(ValLoop):
    """Knowledge Distill loop for validation. It is not only validate student,
    but also validate teacher with the same dataloader.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List]) -> None:
        super().__init__(runner, dataloader, evaluator)
        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model
        assert hasattr(self.model, 'teacher')

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        # compute metrics
        metrics_s = self.evaluator.evaluate(len(self.dataloader.dataset))
        for key, value in metrics_s.items():
            self.runner.message_hub.update_scalar(f'val_student/{key}', value)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter_teacher(idx, data_batch)
        # compute metrics
        metrics_t = self.evaluator.evaluate(len(self.dataloader.dataset))
        for key, value in metrics_t.items():
            self.runner.message_hub.update_scalar(f'val_teacher/{key}', value)

        self.runner.call_hook('after_val_epoch', metrics=None)
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter_teacher(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        outputs = self.model.teacher(data_batch)
        self.evaluator.process(data_batch, outputs)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
