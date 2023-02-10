# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


@HOOKS.register_module()
class DistillationLossDetachHook(Hook):

    priority = 'LOW'

    def __init__(self, detach_epoch) -> None:
        self.detach_epoch = detach_epoch

    def before_train_epoch(self, runner) -> None:
        if runner.epoch == self.detach_epoch:
            model = runner.model
            # TODO: refactor after mmengine using model wrapper
            if is_model_wrapper(model):
                model = model.module
            assert hasattr(model, 'distill_loss_detach')

            runner.logger.info('Distillation stop now!')
            model.distill_loss_detach = True
