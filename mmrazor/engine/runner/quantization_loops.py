# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                   enable_observer)
from torch.nn.intrinsic.qat import freeze_bn_stats
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS


@LOOPS.register_module()
class QATEpochBasedLoop(EpochBasedTrainLoop):
    """`EpochBasedLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        disable_observer_begin (int): The number of total epochs to update
            observers.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            disable_observer_begin: int = 3,
            freeze_bn_begin: int = 3,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)

        self.disable_observer_begin = disable_observer_begin
        self.freeze_bn_begin = freeze_bn_begin

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs:
            # state: observer_enabled, fakequant_enabled
            self.runner.model.apply(enable_fake_quant)
            self.runner.model.apply(enable_observer)
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                # observer disabled during evaluation
                self.runner.model.apply(enable_fake_quant)
                self.runner.model.apply(disable_observer)
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        # TODO freeze bn
        if self._epoch >= self.disable_observer_begin:
            self.runner.model.apply(disable_observer)

        if self._epoch >= self.freeze_bn_begin:
            self.runner.model.apply(freeze_bn_stats)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class QATValLoop(ValLoop):
    """`ValLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
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
            assert hasattr(self.runner.model.module, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.module.data_preprocessor
            self.architecture = self.runner.model.module.architecture
            self.architecture.data_preprocessor = data_preprocessor

        else:
            assert hasattr(self.runner.model, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.data_preprocessor
            self.architecture = self.runner.model.architecture
            self.architecture.data_preprocessor = data_preprocessor

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.runner.model)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        qat_metrics = dict()
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[qat_key] = value
            self.runner.message_hub.log_scalars.pop(f'val/{ori_key}', None)

        self.runner.call_hook('after_val_epoch', metrics=qat_metrics)

        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.architecture)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        qat_metrics = dict()
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[ori_key] = value
            self.runner.message_hub.log_scalars.pop(f'val/{qat_key}', None)

        self.runner.call_hook('after_val_epoch', metrics=qat_metrics)

        self.runner.call_hook('after_val')
        return qat_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], model):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement

        outputs = model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class PTQLoop(TestLoop):
    """`TestLoop` for Post Training Quantization.

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool, optional): Enable FP16 training mode. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 num_cali_batch=32):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.num_cali_batch = num_cali_batch

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        self.runner.model.apply(enable_fake_quant)
        self.runner.model.apply(enable_observer)

        for idx, data_batch in enumerate(self.dataloader):
            if idx == self.num_cali_batch:
                break
            self.run_iter(idx, data_batch)

        self.runner.model.sync_qparams('tensor')
        self.runner.call_hook('after_test_epoch', metrics=None)
        self.runner.call_hook('after_test')

        # todo: hard code to save checkpoint on disk
        self.runner.save_checkpoint(
            self.runner.work_dir,
            'checkpoint_after_ptq.pth',
            file_client_args=None,
            save_optimizer=False,
            save_param_scheduler=False)

        self.runner.model.apply(enable_fake_quant)
        self.runner.model.apply(disable_observer)

        return self.runner.val_loop.run()

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        _ = self.runner.model.calibrate_step(data_batch)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=None)
