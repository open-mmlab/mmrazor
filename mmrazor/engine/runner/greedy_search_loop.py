# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmengine import fileio
from mmengine.evaluator import Evaluator
from mmengine.runner import BaseLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmrazor.models.task_modules import ResourceEstimator
from mmrazor.registry import LOOPS
from mmrazor.structures import export_fix_subnet
from .utils import get_subnet_flops


@LOOPS.register_module()
class GreedySearchLoop(BaseLoop):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 target_flops: Tuple[float],
                 resource_estimator_cfg: Dict[str, Any] = dict(),
                 score_key: str = 'accuracy_top-1',
                 resume_from: Optional[str] = None):
        super().__init__(runner, dataloader)
        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore

        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
        else:
            warnings.warn(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.')

        self.target_flops = sorted(target_flops, reverse=True)
        self.score_key = score_key
        self.resume_from = resume_from
        self.estimator = ResourceEstimator(**resource_estimator_cfg)

        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model

        assert hasattr(self.model, 'mutator')
        self.candidate_choices = {}
        for unit in self.model.mutator.mutable_units:
            self.candidate_choices[unit.name] = unit.candidate_choices
        self.max_subnet = {}
        for unit_name, candidate_choices in self.candidate_choices.items():
            self.max_subnet[unit_name] = len(candidate_choices)
        self.current_subnet = self.max_subnet

        current_subnet_choices = self._channel_bins2choices(
            self.current_subnet)
        self.current_flops = get_subnet_flops(self.model,
                                              current_subnet_choices,
                                              self.estimator)

        self.searched_subnet: List[Dict[str, int]] = []
        self.searched_subnet_flops: List[float] = []

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')

        if self.resume_from:
            self._resume()

        for target in self.target_flops:
            if self.resume_from and self.current_flops <= target:
                continue

            if self.current_flops <= target:
                self.searched_subnet.append(self.current_subnet)
                self.searched_subnet_flops.append(self.current_flops)
                self.runner.logger.info(
                    f'Find model flops {self.current_flops} <= {target}')
                continue

            while self.current_flops > target:
                best_score, best_subnet = None, None

                for unit_name in sorted(self.current_subnet.keys()):
                    if self.current_subnet[unit_name] == 1:
                        # The number of channel_bin has reached the minimum
                        # value
                        continue
                    pruned_subnet = copy.deepcopy(self.current_subnet)
                    pruned_subnet[unit_name] -= 1
                    pruned_subnet_choices = self._channel_bins2choices(
                        pruned_subnet)
                    self.model.set_subnet(pruned_subnet_choices)
                    metrics = self._val_subnet()
                    score = metrics[self.score_key] \
                        if len(metrics) != 0 else 0.
                    self.runner.logger.info(
                        f'Slimming unit {unit_name}, {self.score_key}: {score}'
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_subnet = pruned_subnet

                if best_subnet is None:
                    raise RuntimeError(
                        'Cannot find any valid model, check your '
                        'configurations.')

                self.current_subnet = best_subnet
                current_subnet_choices = self._channel_bins2choices(
                    self.current_subnet)
                self.current_flops = get_subnet_flops(self.model,
                                                      current_subnet_choices,
                                                      self.estimator)
                self.runner.logger.info(
                    f'Greedily find model, score: {self.current_subnet}, '
                    f'FLOPS: {self.current_flops}')
                self._save_searcher_ckpt()

            self.searched_subnet.append(self.current_subnet)
            self.searched_subnet_flops.append(self.current_flops)
            self.runner.logger.info(
                f'Find model flops {self.current_flops} <= {target}')

        self._save_searched_subnet()
        self.runner.call_hook('after_train')

    def _channel_bins2choices(self, subnet_channel_bins):
        choices = {}
        for unit_name, bins in subnet_channel_bins.items():
            # `bins` is in range [1, max_bins]
            choices[unit_name] = self.candidate_choices[unit_name][bins - 1]
        return choices

    @torch.no_grad()
    def _val_subnet(self) -> Dict:
        """Run validation."""
        self.runner.model.eval()
        for data_batch in self.dataloader:
            outputs = self.runner.model.val_step(data_batch)
            self.evaluator.process(data_batch, outputs)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        return metrics

    def _save_searcher_ckpt(self) -> None:
        """Save searcher ckpt, which is different from common ckpt.

        It mainly contains the candicate pool, the top-k candicates with scores
        and the current epoch.
        """
        if self.runner.rank != 0:
            return
        save_for_resume = dict()
        for k in [
                'current_subnet', 'current_flops', 'searched_subnet',
                'searched_subnet_flops'
        ]:
            save_for_resume[k] = getattr(self, k)
        fileio.dump(save_for_resume,
                    osp.join(self.runner.work_dir, 'latest.pkl'))
        self.runner.logger.info(
            f'{len(self.searched_subnet)} subnets have been searched, '
            f'FLOPs are {self.searched_subnet_flops}')

    def _save_searched_subnet(self):
        if self.runner.rank != 0:
            return
        self.runner.logger.info('Search finished:')
        for subnet, flops in zip(self.searched_subnet,
                                 self.searched_subnet_flops):
            subnet_choice = self._channel_bins2choices(subnet)
            self.model.set_subnet(subnet_choice)
            fixed_subnet = export_fix_subnet(self.model)
            save_name = 'FLOPS_{:.2f}M.yaml'.format(flops)
            fileio.dump(fixed_subnet, osp.join(self.runner.work_dir,
                                               save_name))
            self.runner.logger.info(
                f'{save_name} is saved in {self.runner.work_dir}.')

    def _resume(self):
        """Resume searching."""
        searcher_resume = fileio.load(self.resume_from)
        for key, val in searcher_resume.items():
            setattr(self, key, val)
        self.runner.logger.info('#' * 100)
        self.runner.logger.info(f'Current channel_bins dict: '
                                f'{self.current_subnet}, \n'
                                f'Current flops: {self.current_flops}')
        self.runner.logger.info('#' * 100)
