# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence, Union

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import BaseLoop

from mmrazor.models.task_modules import LossWeightScheduler
from mmrazor.registry import HOOKS, PARAM_SCHEDULERS

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class LossWeightSchedulerHook(Hook):

    priority = 'LOW'
    milestones: list = []

    def before_run(self, runner) -> None:

        def build_loss_weight_scheduler(scheduler):
            if not isinstance(scheduler, Sequence):
                schedulers = [scheduler]
            else:
                schedulers = scheduler

            loss_weight_schedulers = []
            for scheduler in schedulers:
                if isinstance(scheduler, LossWeightScheduler):
                    loss_weight_schedulers.append(scheduler)
                elif isinstance(scheduler, dict):
                    _scheduler = copy.deepcopy(scheduler)

                    # Set default end
                    if isinstance(runner.train_loop, BaseLoop):
                        default_end = runner.max_epochs if _scheduler.get(
                            'by_epoch', True) else runner.max_iters
                        _scheduler.setdefault('end', default_end)
                        runner.logger.debug(
                            f'The `end` of {_scheduler["type"]} is not set. '
                            'Use the max epochs/iters of train loop as '
                            'default.')

                    loss_weight_schedulers.append(
                        PARAM_SCHEDULERS.build(
                            _scheduler,
                            default_args=dict(
                                epoch_length=len(runner.train_dataloader))))
                else:
                    raise TypeError(
                        'scheduler should be a LossWeightScheduler object or '
                        f'dict, but got {scheduler}')

            return loss_weight_schedulers

        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        assert hasattr(model, 'loss_weight_scheduler_manager')

        if model.loss_weight_scheduler_manager is None:
            # no specific loss weight scheduler
            return

        schedulers = model.loss_weight_scheduler_manager.schedulers
        model.loss_weight_scheduler_manager.schedulers = \
            build_loss_weight_scheduler(schedulers)

        intervals = []
        epoch_length = len(runner.train_dataloader)
        for scheduler in model.loss_weight_scheduler_manager.schedulers:
            if scheduler.by_epoch:
                intervals.append((scheduler.begin * epoch_length,
                                  scheduler.end * epoch_length))
            else:
                intervals.append((scheduler.begin, scheduler.end))
        # 按照begin排序，按照end构建milestone（如果是by_epoch需要转化成iterbased）
        # 如果当前iter在milestone里，则需要改base value
        intervals = sorted(intervals, key=lambda x: x[0])
        for begin, end in intervals:
            if not self.milestones:
                self.milestones.append(end)
            elif end > self.milestones[-1]:
                self.milestones.append(end)

    def set_loss_weight_multiplier(self, runner, scheduler_manager, by_epoch):
        schedulers = scheduler_manager.schedulers
        assert isinstance(schedulers, list)
        for scheduler in schedulers:
            if scheduler.by_epoch == by_epoch:
                cur_iter = runner.iter
                cur_epoch = runner.epoch
                if cur_iter in self.milestones:
                    # move to the next stage and modify the base value
                    scheduler_manager.base_value = scheduler_manager.cur_value

                base_value = scheduler_manager.base_value
                cur_value = scheduler_manager.cur_value
                multiplier = scheduler.get_multiplier(
                    base_value, cur_value, cur_epoch if by_epoch else cur_iter)
                if multiplier is not None:
                    scheduler_manager.cur_value = multiplier
                    break

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None,
                          outputs: Optional[dict] = None) -> None:

        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model

        if model.loss_weight_scheduler_manager is None:
            # no specific loss weight scheduler
            return

        self.set_loss_weight_multiplier(
            runner, model.loss_weight_scheduler_manager, by_epoch=False)

    def before_train_epoch(self, runner) -> None:
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model

        if model.loss_weight_scheduler_manager is None:
            # no specific loss weight scheduler
            return

        self.set_loss_weight_multiplier(
            runner, model.loss_weight_scheduler_manager, by_epoch=True)
