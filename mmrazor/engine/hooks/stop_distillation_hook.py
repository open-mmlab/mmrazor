# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


@HOOKS.register_module()
class StopDistillHook(Hook):
    """Stop distilling at a certain time.

    Args:
        stop_epoch (int): Stop distillation at this epoch.
    """

    priority = 'LOW'

    def __init__(self, stop_epoch: int) -> None:
        self.stop_epoch = stop_epoch

    def before_train_epoch(self, runner) -> None:
        """Stop distillation."""
        if runner.epoch >= self.stop_epoch:
            model = runner.model
            # TODO: refactor after mmengine using model wrapper
            if is_model_wrapper(model):
                model = model.module
            assert hasattr(model, 'distillation_stopped')

            runner.logger.info('Distillation has been stopped!')
            model.distillation_stopped = True
