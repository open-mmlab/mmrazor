# Copyright (c) OpenMMLab. All rights reserved.
import io
import os.path as osp
import warnings

import torch
from mmcv.fileio import FileClient
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook, master_only


@HOOKS.register_module()
class LossRecHook(Hook):

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 file_client_args=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.file_client_args = file_client_args

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        runner.logger.info((f'Checkpoints will be saved to {self.out_dir} by '
                            f'{self.file_client.name}.'))

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    ('create_symlink is set as True by the user but is changed'
                     'to be False because creating symbolic link is not '
                     f'allowed in {self.file_client.name}'))
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save loss records for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving loss records at {runner.epoch + 1} epochs')
            self._save_loss_rec(runner)

    @master_only
    def _save_loss_rec(self, runner):
        filename = 'loss_rec_epoch_{}.pth'.format(runner.epoch + 1) \
            if self.by_epoch \
            else 'loss_rec_iter_{}.pth'.format(runner.iter + 1)
        filepath = osp.join(self.out_dir, filename)

        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, 'loss_rec')

        with io.BytesIO() as f:
            torch.save(model.loss_rec, f)
            self.file_client.put(f.getvalue(), filepath)

        if runner.meta is not None:
            if self.by_epoch:
                cur_loss_rec_filename = self.args.get(
                    'filename_tmpl',
                    'loss_rec_epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_loss_rec_filename = self.args.get(
                    'filename_tmpl',
                    'loss_rec_iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs'][
                'last_loss_rec'] = self.file_client.join_path(
                    self.out_dir, cur_loss_rec_filename)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'loss_rec_epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'loss_rec_iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save loss records for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving loss records at {runner.iter + 1} iterations')
            self._save_loss_rec(runner)
