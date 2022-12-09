# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from mmengine.evaluator import Evaluator
from mmengine.runner import TestLoop
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS
from .mixins import CalibrateBNMixin


@LOOPS.register_module()
class AutoSlimTestLoop(TestLoop, CalibrateBNMixin):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 calibrate_sample_num: int = 4096) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        # just for convenience
        self._model = model
        self.calibrate_sample_num = calibrate_sample_num

    def run(self) -> None:
        """Launch validation."""
        self.runner.call_hook('before_test')

        all_metrics = dict()

        self._model.set_max_subnet()
        self.calibrate_bn_statistics(self.runner.train_dataloader,
                                     self.calibrate_sample_num)
        metrics = self._evaluate_once()
        all_metrics.update(add_prefix(metrics, 'max_subnet'))

        self._model.set_min_subnet()
        self.calibrate_bn_statistics(self.runner.train_dataloader,
                                     self.calibrate_sample_num)
        metrics = self._evaluate_once()
        all_metrics.update(add_prefix(metrics, 'min_subnet'))

        self.runner.call_hook('after_test_epoch', metrics=all_metrics)
        self.runner.call_hook('after_test')

    def _evaluate_once(self) -> Dict:
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        return self.evaluator.evaluate(len(self.dataloader.dataset))
