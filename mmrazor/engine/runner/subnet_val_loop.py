# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS, TASK_UTILS
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
        fix_subnet_kind (str): fix subnet kinds when evaluate, this would be
            `sample_kinds` if not specified
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to 4096.
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to dict(type='mmrazor.ResourceEstimator').
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
        fix_subnet_kinds: List[str] = [],
        calibrate_sample_num: int = 4096,
        estimator_cfg: Optional[Dict] = dict(type='mmrazor.ResourceEstimator')
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        self.model = model
        if len(fix_subnet_kinds) == 0 and not hasattr(self.model,
                                                      'sample_kinds'):
            raise ValueError(
                'neither fix_subnet_kinds nor self.model.sample_kinds exists')

        self.evaluate_kinds = fix_subnet_kinds if len(
            fix_subnet_kinds) > 0 else getattr(self.model, 'sample_kinds')

        self.calibrate_sample_num = calibrate_sample_num
        self.estimator = None
        if estimator_cfg:
            self.estimator = TASK_UTILS.build(estimator_cfg)

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        all_metrics = dict()

        for kind in self.evaluate_kinds:
            if kind == 'max':
                self.model.mutator.set_max_choices()
            elif kind == 'min':
                self.model.mutator.set_min_choices()
            elif 'random' in kind:
                self.model.mutator.set_choices(
                    self.model.mutator.sample_choices())
            else:
                raise NotImplementedError(f'Unsupported Subnet {kind}')

            metrics = self._evaluate_once()
            all_metrics.update(add_prefix(metrics, f'{kind}_subnet'))

        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')

    def _evaluate_once(self) -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        self.calibrate_bn_statistics(self.runner.train_dataloader,
                                     self.calibrate_sample_num)
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        if self.estimator:
            resource_metrics = self.estimator.estimate(self.model)
            metrics.update(resource_metrics)

        return metrics
