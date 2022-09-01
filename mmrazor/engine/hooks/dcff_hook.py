# Copyright (c) OpenMMLab. All rights reserved.
import math
import time
from typing import Dict, Optional, Sequence

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class DCFFHook(Hook):
    """DCFF Hook using MME Hook with single and multi GPU test.

    Args:
        save_model_name (str): Save the file name of the compactmodel.
        dcff_count (int): Every dcff_cout epochs or iters calculates
            the importance of filters. Default: 1
    """

    def __init__(self,
                 by_epoch: bool = True,
                 save_model_name='model_best_compact.pth',
                 dcff_count=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_model_name = save_model_name
        self.dcff_count = dcff_count
        self.by_epoch = by_epoch

    def before_run(self, runner):
        super().before_run(runner)
        self.algorithm = getattr(runner.model, 'module', runner.model)
        self.mutator = getattr(self.algorithm, 'mutator')
        # start epoch and iter before train.
        self.start_epoch = runner.epoch
        self.start_iter = runner.iter
        # Get current device: GPU or CPU.
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        if not self.by_epoch and runner.iter % self.dcff_count == 0:
            runner.logger.info(
                f"Computing iter {runner.iter+1}'s all fused layer...")
            stime = time.time()
            t = self._calc_temperature(runner.iter, runner.max_iters)
            self.mutator.calc_information(t, runner.iter, self.start_iter)
            # no runner.meta, so calculate kl layer in calc_information()
            runner.logger.info(
                f'Iter {runner.iter+1}' +
                f' fuse all layers cost: {time.time()-stime:.2f}s')
        super().before_train_iter(runner, batch_idx, data_batch)

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        if self.by_epoch and runner.iter % self.dcff_count == 0:
            runner.logger.info(
                f"Computing epoch {runner.epoch+1}'s all fused layer...")
            stime = time.time()
            # print("before kl clac")
            t = self._calc_temperature(runner.epoch, runner.max_epochs)
            self.mutator.calc_information(t, runner.epoch, self.start_epoch)
            # print("after kl clac")
            # no runner.meta, so calculate kl layer in calc_information()
            runner.logger.info(
                f'Epoch {runner.epoch+1}' +
                f' fuse all layers cost: {time.time()-stime:.2f}s')
        super().before_train_epoch(runner)

    def _calc_temperature(self, cur_num, max_num):
        """Calculate temperature param."""
        # Set the fixed parameters required to calculate the temperature t
        t_s, t_e, k = 1, 10000, 1
        # e is the current training epoch
        # E is the total number of training epochs
        e = cur_num
        E = max_num

        A = 2 * (t_e - t_s) * (1 + math.exp(-k * E)) / (1 - math.exp(-k * E))
        T = A / (1 + math.exp(-k * e)) + t_s - A / 2
        t = 1 / T
        return t
