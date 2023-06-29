# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.logging import MessageHub
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


@HOOKS.register_module()
class StopDistillHook(Hook):
    """Stop distilling at a certain time.

    Args:
        stop_epoch (int): Stop distillation at this epoch.
    """

    def __init__(self, stop_epoch: int) -> None:
        self.stop_epoch = stop_epoch

    def _clear_message_hub(self):
        """Private method to clear distillation-related log scalars."""
        message_hub = MessageHub.get_current_instance()
        log_scalars = message_hub.log_scalars
        keys_del = [key for key in log_scalars.keys() if 'distill' in key]
        for key in keys_del:
            del log_scalars[key]

    def before_train_epoch(self, runner) -> None:
        """Stop distillation."""
        if runner.epoch == self.stop_epoch:
            model = runner.model
            # TODO: refactor after mmengine using model wrapper
            if is_model_wrapper(model):
                model = model.module
            assert hasattr(model, 'distillation_stopped')

            runner.logger.info('Distillation has been stopped!')
            model.distillation_stopped[0] = True

            self._clear_message_hub()
