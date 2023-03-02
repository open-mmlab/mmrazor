# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

from mmengine.dist import master_only
from mmengine.fileio import FileClient, dump
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.structures import convert_fix_subnet, export_fix_subnet

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class DumpSubnetHook(Hook):
    """Dump subnet periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        out_dir (str, optional | Path): The root directory to save checkpoints.
            If not specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``. For example,
            if the input ``our_dir`` is ``./tmp`` and ``runner.work_dir`` is
            ``./work_dir/cur_exp``, then the ckpt will be saved in
            ``./tmp/cur_exp``. Defaults to None.
        max_keep_subnets (int): The maximum subnets to keep.
            In some cases we want only the latest few subnets and would
            like to delete old ones to save the disk space.
            Defaults to -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be
            saved regardless of interval. Defaults to True.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Defaults to None.
    """
    out_dir: str

    priority = 'VERY_LOW'

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_subnets: int = -1,
                 save_last: bool = True,
                 file_client_args: Optional[dict] = None,
                 **kwargs) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.out_dir = out_dir  # type: ignore
        self.max_keep_subnets = max_keep_subnets
        self.save_last = save_last
        self.args = kwargs
        self.file_client_args = file_client_args

    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)
        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(
                self.out_dir, basename)  # type: ignore  # noqa: E501

        runner.logger.info(f'Subnets will be saved to {self.out_dir} by '
                           f'{self.file_client.name}.')

    def after_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # save subnet for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval) or (
                self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(f'Saving subnet at {runner.epoch + 1} epochs')
            self._save_subnet(runner)

    @master_only
    def _save_subnet(self, runner) -> None:
        """Save the current best subnet.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model.module if runner.distributed else runner.model

        # delete non-leaf tensor to get deepcopy(model).
        # TODO solve the hard case.
        for module in model.architecture.modules():
            if isinstance(module, BaseMutable):
                if hasattr(module, 'arch_weights'):
                    delattr(module, 'arch_weights')

        copied_model = copy.deepcopy(model)
        copied_model.mutator.set_choices(copied_model.mutator.sample_choices())

        subnet_dict = export_fix_subnet(copied_model)[0]
        subnet_dict = convert_fix_subnet(subnet_dict)

        if self.by_epoch:
            subnet_filename = self.args.get(
                'filename_tmpl',
                'subnet_epoch_{}.yaml').format(runner.epoch + 1)
        else:
            subnet_filename = self.args.get(
                'filename_tmpl', 'subnet_iter_{}.yaml').format(runner.iter + 1)

        file_client = FileClient.infer_client(self.file_client_args,
                                              self.out_dir)
        filepath = file_client.join_path(self.out_dir, subnet_filename)

        dump(subnet_dict, filepath, file_format='yaml')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        """Save the subnet after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_train_iters(runner, self.interval) or \
                (self.save_last and
                 self.is_last_train_iter(runner)):
            runner.logger.info(
                f'Saving subnet at {runner.iter + 1} iterations')
            self._save_subnet(runner)
