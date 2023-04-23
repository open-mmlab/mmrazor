# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer)
    from torch.nn.intrinsic.qat import freeze_bn_stats
except ImportError:
    from mmrazor.utils import get_placeholder

    disable_observer = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    freeze_bn_stats = get_placeholder('torch>=1.13')

from mmengine.dist import all_reduce_params, is_distributed
from torch.utils.data import DataLoader

from mmrazor.models import register_torch_fake_quants, register_torch_observers
from mmrazor.models.fake_quants import (enable_param_learning,
                                        enable_static_estimate, enable_val)
from mmrazor.registry import LOOPS

TORCH_observers = register_torch_observers()
TORCH_fake_quants = register_torch_fake_quants()


@LOOPS.register_module()
class QATEpochBasedLoop(EpochBasedTrainLoop):
    """`EpochBasedLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        disable_observer_begin (int): The number of total epochs to update
            observers. Defaults to -1, which means observers are enabled
            all the time.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats. Defaults to -1, which means no need to freeze bn.
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
            disable_observer_begin: int = -1,
            freeze_bn_begin: int = -1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)

        self.disable_observer_begin = disable_observer_begin
        self.freeze_bn_begin = freeze_bn_begin

    def prepare_for_run_epoch(self):
        """Toggle the state of the observers and fake quantizers before qat
        training."""
        self.runner.model.apply(enable_fake_quant)

        # The initialized _epoch equals to 0 so _epoch + 1
        # equal to the current epoch
        if (self.disable_observer_begin > 0
                and self._epoch + 1 >= self.disable_observer_begin):
            self.runner.model.apply(disable_observer)
        else:
            self.runner.model.apply(enable_observer)

        if (self.freeze_bn_begin > 0
                and self._epoch + 1 >= self.freeze_bn_begin):
            self.runner.model.apply(freeze_bn_stats)

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_fake_quant)
        self.runner.model.apply(disable_observer)

    def run(self):
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs:
            self.prepare_for_run_epoch()
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.model.sync_qparams(src_mode='loss')
        # Make sure the registered buffer such as `observer_enabled` is
        # correct in the saved checkpoint.
        self.prepare_for_val()
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class LSQEpochBasedLoop(QATEpochBasedLoop):
    """`EpochBasedLoop` for `LEARNED STEP SIZE QUANTIZATION`

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats. Defaults to -1, which means no need to freeze bn.
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
            freeze_bn_begin: int = -1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            freeze_bn_begin=freeze_bn_begin,
            dynamic_intervals=dynamic_intervals)

        self.is_first_batch = True
        self.distributed = is_distributed()

    def prepare_for_run_epoch(self):
        """Toggle the state of the observers and fake quantizers before qat
        training."""
        if (self.freeze_bn_begin > 0
                and self._epoch + 1 >= self.freeze_bn_begin):
            self.runner.model.apply(freeze_bn_stats)

        self.runner.model.apply(enable_param_learning)

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            if self.is_first_batch:
                # lsq observer init
                self.runner.model.apply(enable_static_estimate)

            self.run_iter(idx, data_batch)

            if self.is_first_batch:
                # In the first batch, scale in LearnableFakeQuantize is
                # calculated through lsq observer. As the values of `scale` of
                # different observers in different rank are usually different,
                # we have to sync the `scale` here.
                if self.distributed:
                    all_reduce_params(
                        self.runner.model.parameters(), op='mean')

                # Change back to param learning mode
                self.is_first_batch = False
                self.runner.model.apply(enable_param_learning)

        self.runner.model.sync_qparams(src_mode='loss')
        # Make sure the registered buffer such as `observer_enabled` is
        # correct in the saved checkpoint.
        self.prepare_for_val()
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
                 calibrate_dataloader: Union[DataLoader, Dict],
                 calibrate_steps=32,
                 fp16: bool = False,
                 only_val=False):
        super().__init__(runner, dataloader, evaluator, fp16)
        if isinstance(calibrate_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.calibrate_dataloader = runner.build_dataloader(
                calibrate_dataloader,
                seed=runner.seed,
                diff_rank_seed=diff_rank_seed)
        else:
            self.calibrate_dataloader = calibrate_dataloader

        self.calibrate_steps = calibrate_steps
        self.only_val = only_val

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        self.runner.model.eval()

        if not self.only_val:
            self.runner.model.apply(enable_fake_quant)
            self.runner.model.apply(enable_observer)

            print_log('Star calibratiion...')
            for idx, data_batch in enumerate(self.calibrate_dataloader):
                if idx == self.calibrate_steps:
                    break
                self.run_iter(idx, data_batch)
            print_log('Finish calibratiion!')

            self.runner.model.apply(enable_fake_quant)
            self.runner.model.apply(disable_observer)

            save_dir = os.path.join(self.runner.work_dir,
                                    self.runner.timestamp)
            self.runner.save_checkpoint(
                save_dir,
                'model_ptq.pth',
                file_client_args=None,
                save_optimizer=False,
                save_param_scheduler=False)
            print_log(f'Quantized model is saved in {save_dir}')

        print_log('Start Evaluating quantized model...')
        self.runner.model.apply(enable_fake_quant)
        self.runner.model.apply(disable_observer)
        metricts = self.runner.val_loop.run()
        self.runner.call_hook('after_test_epoch', metrics=metricts)
        self.runner.call_hook('after_test')

        return metricts

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
