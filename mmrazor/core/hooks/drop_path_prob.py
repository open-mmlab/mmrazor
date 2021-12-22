# Copyright (c) Open-MMLab. All rights reserved.

from mmcv.cnn.bricks import DropPath
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DropPathProbHook(Hook):
    """Set drop_path_prob periodically.

    Args:
        max_prob (float): The max probability of dropping.
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
    """

    def __init__(self, max_prob, interval=-1, by_epoch=True, **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.max_prob = max_prob
        assert self.by_epoch

    def before_train_epoch(self, runner):
        """Executed in before_train_epoch stage."""
        if not self.by_epoch:
            return

        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            cur_epoch = runner.epoch
            max_epoch = runner._max_epochs
            drop_prob = self.max_prob * (cur_epoch * 1.0) / (max_epoch * 1.0)
            for module in runner.model.modules():
                if isinstance(module, DropPath):
                    module.drop_prob = drop_prob
            runner.logger.info(f'Set drop_prob to {drop_prob} \
                    at {runner.epoch + 1} epochs')
