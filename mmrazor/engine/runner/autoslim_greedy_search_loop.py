# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmengine import fileio
from mmengine.evaluator import Evaluator
from mmengine.runner import TestLoop
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS, TASK_UTILS
from mmrazor.structures import convert_fix_subnet, export_fix_subnet
from .utils import check_subnet_resources


@LOOPS.register_module()
class AutoSlimGreedySearchLoop(TestLoop):
    """Loop for Greedy searching in AutoSlim. Please refer to
    https://arxiv.org/abs/1903.11728 for more details.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        target_flops (Tuple[float]): The FLOPs limitation of target subnets.
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to None.
        score_key (str): Specify one metric in evaluation results to score
            candidates. Defaults to 'accuracy_top-1'.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 target_flops: Tuple[float],
                 estimator_cfg: Dict[str, Any] = dict(),
                 score_key: str = 'accuracy/top1',
                 resume_from: Optional[str] = None):
        super().__init__(runner, dataloader, evaluator)

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

        # initialize estimator
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)

        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model

        assert hasattr(self.model, 'mutator')
        units = self.model.mutator.mutable_units

        self.candidate_choices = {}
        for unit in units:
            self.candidate_choices[unit.alias] = unit.candidate_choices

        self.max_subnet = {}
        for name, candidate_choices in self.candidate_choices.items():
            self.max_subnet[name] = len(candidate_choices)
        self.current_subnet = self.max_subnet

        current_subnet_choices = self._channel_bins2choices(
            self.current_subnet)
        _, results = check_subnet_resources(self.model, current_subnet_choices,
                                            self.estimator)
        self.current_flops = results['flops']

        self.searched_subnet: List[Dict[str, int]] = []
        self.searched_subnet_flops: List[float] = []

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_test')

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
                    self.model.mutator.set_choices(pruned_subnet_choices)
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
                _, results = check_subnet_resources(self.model,
                                                    current_subnet_choices,
                                                    self.estimator)
                self.current_flops = results['flops']
                self.runner.logger.info(
                    f'Greedily find model, score: {best_score}, '
                    f'{self.current_subnet}, FLOPS: {self.current_flops}')
                self._save_searcher_ckpt()

            self.searched_subnet.append(self.current_subnet)
            self.searched_subnet_flops.append(self.current_flops)
            self.runner.logger.info(
                f'Find model flops {self.current_flops} <= {target}')

        self._save_searched_subnet()
        self.runner.call_hook('after_test')

    def _channel_bins2choices(self, subnet_channel_bins):
        """Convert the channel bin number of a channel unit to the choice
        (ratio when choice_mode='ratio' and channel number when
        choice_mode='number')."""
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
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
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
        """Save the final searched subnet dict."""
        if self.runner.rank != 0:
            return
        self.runner.logger.info('Search finished:')
        for subnet, flops in zip(self.searched_subnet,
                                 self.searched_subnet_flops):
            subnet_choice = self._channel_bins2choices(subnet)
            self.model.mutator.set_choices(subnet_choice)
            fixed_subnet, _ = export_fix_subnet(self.model)
            save_name = 'FLOPS_{:.2f}M.yaml'.format(flops)
            fixed_subnet = convert_fix_subnet(fixed_subnet)
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
