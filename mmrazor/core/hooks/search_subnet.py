# Copyright (c) Open-MMLab. All rights reserved.
import os

from mmcv.runner import HOOKS, Hook, master_only


@HOOKS.register_module()
class SearchSubnetHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be saved
            regardless of interval.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch

        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs

    def before_run(self, runner):
        """Executed in before_run stage."""
        if not self.out_dir:
            self.out_dir = runner.work_dir

    def after_train_epoch(self, runner):
        """Executed in after_train_epoch stage."""
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(f'Saving subnet at {runner.epoch + 1} epochs')
            self._search_subnet(runner)

    @master_only
    def _search_subnet(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        runner.search_subnet(self.out_dir, **self.args)
        if runner.meta is not None:
            if self.by_epoch:
                cur_subnet_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.yaml').format(runner.epoch + 1)
            else:
                cur_subnet_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.yaml').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_subnet'] = os.path.join(
                self.out_dir, cur_subnet_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.yaml'
                current_subnet = runner.epoch + 1
            else:
                name = 'iter_{}.yaml'
                current_subnet = runner.iter + 1
            redundant_subnets = range(
                current_subnet - self.max_keep_subnets * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_subnets:
                subnet_path = os.path.join(self.out_dir,
                                           filename_tmpl.format(_step))
                if os.path.exists(subnet_path):
                    os.remove(subnet_path)
                else:
                    break

    def after_train_iter(self, runner):
        """Executed in after_train_iter stage."""
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving subnet at {runner.iter + 1} iterations')

            self._search_subnet(runner)
