# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from mmengine.evaluator import Evaluator
from mmengine.hooks import CheckpointHook
from mmengine.runner import ValLoop
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS
from .utils import CalibrateBNMixin


@LOOPS.register_module()
class SubnetValLoop(ValLoop, CalibrateBNMixin):
    """Loop for subnet validation in NAS with BN re-calibration.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
        evaluate_fixed_subnet (bool): Whether to evaluate a fixed subnet only
            or not. Defaults to False.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to 4096.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 evaluate_fixed_subnet: bool = False,
                 calibrate_sample_num: int = 4096) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        # just for convenience
        self._model = model
        self.evaluate_fixed_subnet = evaluate_fixed_subnet
        self.calibrate_sample_num = calibrate_sample_num

        # remove CheckpointHook to avoid extra problems.
        for hook in self.runner._hooks:
            if isinstance(hook, CheckpointHook):
                self.runner._hooks.remove(hook)
                break

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        all_metrics = dict()

        # sample subnet by mutator
        # for scale in ['a_max', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
        #     self._model.set_attentivenas_subnet(scale)
        #     metrics = self._evaluate_once()
        #     all_metrics.update(add_prefix(metrics, scale))

        if self.evaluate_fixed_subnet:
            metrics = self._evaluate_once()
            all_metrics.update(add_prefix(metrics, 'fix_subnet'))
        else:
            self._model.set_max_subnet()
            metrics = self._evaluate_once()
            all_metrics.update(add_prefix(metrics, 'max_subnet'))

            self._model.set_min_subnet()
            metrics = self._evaluate_once()
            all_metrics.update(add_prefix(metrics, 'min_subnet'))

            sample_nums = self._model.random_samples if hasattr(
                self._model, 'random_samples') else self._model.samples
            for subnet_idx in range(sample_nums):
                self._model.set_subnet(self._model.sample_subnet())
                # compute student metrics
                metrics = self._evaluate_once()
                all_metrics.update(
                    add_prefix(metrics, f'random_subnet_{subnet_idx}'))

        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')

    def _evaluate_once(self) -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        # self.calibrate_bn_statistics(self.runner.train_dataloader,
        #                              self.calibrate_sample_num)
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        return self.evaluator.evaluate(len(self.dataloader.dataset))
