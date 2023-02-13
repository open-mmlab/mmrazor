# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop, autocast
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
            assert hasattr(self.runner.model.module, 'teacher')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.module.data_preprocessor
            self.teacher = self.runner.model.module.teacher
            self.teacher.data_preprocessor = data_preprocessor

        else:
            assert hasattr(self.runner.model, 'teacher')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.data_preprocessor
            self.teacher = self.runner.model.teacher
            self.teacher.data_preprocessor = data_preprocessor

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        # compute student metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('before_val_epoch')
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter_teacher(idx, data_batch)
        # compute teacher metrics
        teacher_metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        for key, value in teacher_metrics.items():
            teacher_key = 'teacher.' + key
            metrics[teacher_key] = value

        self.runner.call_hook('after_val_epoch', metrics=metrics)
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

        with autocast(enabled=self.fp16):
            # outputs should be sequence of BaseDataElement
            outputs = self.teacher.val_step(data_batch)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class SelfDistillValLoop(ValLoop):
    """Knowledge Distill loop for validation. Only validate student.

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
        # compute student metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        student_metrics = dict()
        for key, value in metrics.items():
            student_key = 'student.' + key
            student_metrics[student_key] = value

        self.runner.call_hook('after_val_epoch', metrics=student_metrics)
        self.runner.call_hook('after_val')
