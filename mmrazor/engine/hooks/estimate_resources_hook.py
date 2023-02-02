# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, Optional, Sequence

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.structures import BaseDataElement

from mmrazor.registry import TASK_UTILS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class EstimateResourcesHook(Hook):
    """Estimate model resources periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default to True.
        estimator_cfg (Dict[str, Any]): Used for building a resource estimator.
            Default to None.

    Example:
    >>> add the `EstimatorResourcesHook` in custom_hooks as follows:
        custom_hooks = [
            dict(type='mmrazor.EstimateResourcesHook',
                 interval=1,
                 by_epoch=True,
                 estimator_cfg=dict(input_shape=(1, 3, 64, 64)))
        ]
    """
    out_dir: str

    priority = 'VERY_LOW'

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 estimator_cfg: Dict[str, Any] = None,
                 **kwargs) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Estimate model resources after every n val epochs.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        if self.every_n_epochs(runner, self.interval):
            self.estimate_resources(runner)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence[BaseDataElement]] = None) \
            -> None:
        """Estimate model resources after every n val iters.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            return

        if self.every_n_train_iters(runner, self.interval):
            self.estimate_resources(runner)

    def estimate_resources(self, runner) -> None:
        """Estimate model resources: latency/flops/params."""
        model = runner.model.module if runner.distributed else runner.model

        # TODO confirm the state judgement.
        if hasattr(model, 'is_supernet') and model.is_supernet:
            model = self.export_subnet(model)

        resource_metrics = self.estimator.estimate(model)
        runner.logger.info(f'Estimate model resources: {resource_metrics}')

    def export_subnet(self, model) -> torch.nn.Module:
        """Export current best subnet.

        NOTE: This method is called when it comes to those NAS algorithms that
        require building a supernet for training.

        For those algorithms, measuring subnet resources is more meaningful
        than supernet during validation, therefore this method is required to
        get the current searched subnet from the supernet.
        """
        # Avoid circular import
        from mmrazor.models.mutables.base_mutable import BaseMutable
        from mmrazor.structures import export_fix_subnet, load_fix_subnet

        # delete non-leaf tensor to get deepcopy(model).
        # TODO solve the hard case.
        for module in model.architecture.modules():
            if isinstance(module, BaseMutable):
                if hasattr(module, 'arch_weights'):
                    delattr(module, 'arch_weights')

        copied_model = copy.deepcopy(model)
        copied_model.mutator.set_choices(copied_model.mutator.sample_choices())

        subnet_dict = export_fix_subnet(copied_model)[0]
        load_fix_subnet(copied_model, subnet_dict)

        return copied_model
