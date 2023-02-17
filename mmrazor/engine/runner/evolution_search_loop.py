# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine import fileio
from mmengine.dist import broadcast_object_list
from mmengine.evaluator import Evaluator
from mmengine.runner import EpochBasedTrainLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS, TASK_UTILS
from mmrazor.structures import (Candidates, convert_fix_subnet,
                                export_fix_subnet)
from mmrazor.utils import SupportRandomSubnet
from .utils import CalibrateBNMixin, check_subnet_resources, crossover


@LOOPS.register_module()
class EvolutionSearchLoop(EpochBasedTrainLoop, CalibrateBNMixin):
    """Loop for evolution searching.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        max_epochs (int): Total searching epochs. Defaults to 20.
        max_keep_ckpts (int): The maximum checkpoints of searcher to keep.
            Defaults to 3.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching.
        num_candidates (int): The length of candidate pool. Defaults to 50.
        top_k (int): Specify top k candidates based on scores. Defaults to 10.
        num_mutation (int): The number of candidates got by mutation.
            Defaults to 25.
        num_crossover (int): The number of candidates got by crossover.
            Defaults to 25.
        mutate_prob (float): The probability of mutation. Defaults to 0.1.
        crossover_prob (float): The probability of crossover. Defaults to 0.5.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to -1.
        constraints_range (Dict[str, Any]): Constraints to be used for
            screening candidates. Defaults to dict(flops=(0, 330)).
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to None.
        predictor_cfg (dict, Optional): Used for building a metric predictor.
            Defaults to None.
        score_key (str): Specify one metric in evaluation results to score
            candidates. Defaults to 'accuracy_top-1'.
        init_candidates (str, optional): The candidates file path, which is
            used to init `self.candidates`. Its format is usually in .yaml
            format. Defaults to None.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 max_epochs: int = 20,
                 max_keep_ckpts: int = 3,
                 resume_from: Optional[str] = None,
                 num_candidates: int = 50,
                 top_k: int = 10,
                 num_mutation: int = 25,
                 num_crossover: int = 25,
                 mutate_prob: float = 0.1,
                 crossover_prob: float = 0.5,
                 calibrate_sample_num: int = -1,
                 constraints_range: Dict[str, Any] = dict(flops=(0., 330.)),
                 estimator_cfg: Optional[Dict] = None,
                 predictor_cfg: Optional[Dict] = None,
                 score_key: str = 'accuracy/top1',
                 init_candidates: Optional[str] = None) -> None:
        super().__init__(runner, dataloader, max_epochs)
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

        self.num_candidates = num_candidates
        self.top_k = top_k
        self.constraints_range = constraints_range
        self.calibrate_sample_num = calibrate_sample_num
        self.score_key = score_key
        self.num_mutation = num_mutation
        self.num_crossover = num_crossover
        self.mutate_prob = mutate_prob
        self.crossover_prob = crossover_prob
        self.max_keep_ckpts = max_keep_ckpts
        self.resume_from = resume_from
        self.fp16 = False

        if init_candidates is None:
            self.candidates = Candidates()
        else:
            self.candidates = fileio.load(init_candidates)
            assert isinstance(self.candidates, Candidates), 'please use the \
                correct init candidates file'

        self.top_k_candidates = Candidates()

        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model

        # initialize estimator
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)

        # initialize predictor
        self.use_predictor = False
        self.predictor_cfg = predictor_cfg
        if self.predictor_cfg is not None:
            self.predictor_cfg['score_key'] = self.score_key
            self.predictor_cfg['search_groups'] = \
                self.model.mutator.search_groups
            self.predictor = TASK_UTILS.build(self.predictor_cfg)

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')

        if self.predictor_cfg is not None:
            self._init_predictor()

        if self.resume_from:
            self._resume()

        while self._epoch < self._max_epochs:
            self.run_epoch()
            self._save_searcher_ckpt()

        self._save_best_fix_subnet()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch.

        Steps:
            1. Sample some new candidates from the supernet. Then Append them
                to the candidates, Thus make its number equal to the specified
                number.
            2. Validate these candidates(step 1) and update their scores.
            3. Pick the top k candidates based on the scores(step 2), which
                will be used in mutation and crossover.
            4. Implement Mutation and crossover, generate better candidates.
        """
        self.sample_candidates()
        self.update_candidates_scores()

        scores_before = self.top_k_candidates.scores
        self.runner.logger.info(f'top k scores before update: '
                                f'{scores_before}')

        self.candidates.extend(self.top_k_candidates)
        self.candidates.sort_by(key_indicator='score', reverse=True)
        self.top_k_candidates = Candidates(self.candidates.data[:self.top_k])

        scores_after = self.top_k_candidates.scores
        self.runner.logger.info(f'top k scores after update: '
                                f'{scores_after}')

        mutation_candidates = self.gen_mutation_candidates()
        self.candidates_mutator_crossover = Candidates(mutation_candidates)
        crossover_candidates = self.gen_crossover_candidates()
        self.candidates_mutator_crossover.extend(crossover_candidates)

        assert len(self.candidates_mutator_crossover
                   ) <= self.num_candidates, 'Total of mutation and \
            crossover should be less than the number of candidates.'

        self.candidates = self.candidates_mutator_crossover
        self._epoch += 1

    def sample_candidates(self) -> None:
        """Update candidate pool contains specified number of candicates."""
        candidates_resources = []
        init_candidates = len(self.candidates)
        if self.runner.rank == 0:
            while len(self.candidates) < self.num_candidates:
                candidate = self.model.mutator.sample_choices()
                is_pass, result = self._check_constraints(
                    random_subnet=candidate)
                if is_pass:
                    self.candidates.append(candidate)
                    candidates_resources.append(result)
            self.candidates = Candidates(self.candidates.data)
        else:
            self.candidates = Candidates([dict(a=0)] * self.num_candidates)

        if len(candidates_resources) > 0:
            self.candidates.update_resources(
                candidates_resources,
                start=len(self.candidates.data) - len(candidates_resources))
            assert init_candidates + len(
                candidates_resources) == self.num_candidates

        # broadcast candidates to val with multi-GPUs.
        broadcast_object_list(self.candidates.data)

    def update_candidates_scores(self) -> None:
        """Validate candicate one by one from the candicate pool, and update
        top-k candicates."""
        for i, candidate in enumerate(self.candidates.subnets):
            self.model.mutator.set_choices(candidate)
            metrics = self._val_candidate(use_predictor=self.use_predictor)
            score = round(metrics[self.score_key], 2) \
                if len(metrics) != 0 else 0.
            self.candidates.set_resource(i, score, 'score')
            self.runner.logger.info(
                f'Epoch:[{self._epoch}/{self._max_epochs}] '
                f'Candidate:[{i + 1}/{self.num_candidates}] '
                f'Flops: {self.candidates.resources("flops")[i]} '
                f'Params: {self.candidates.resources("params")[i]} '
                f'Latency: {self.candidates.resources("latency")[i]} '
                f'Score: {self.candidates.scores[i]} ')

    def gen_mutation_candidates(self):
        """Generate specified number of mutation candicates."""
        mutation_resources = []
        mutation_candidates: List = []
        max_mutate_iters = self.num_mutation * 10
        mutate_iter = 0
        while len(mutation_candidates) < self.num_mutation:
            mutate_iter += 1
            if mutate_iter > max_mutate_iters:
                break

            mutation_candidate = self._mutation()

            is_pass, result = self._check_constraints(
                random_subnet=mutation_candidate)
            if is_pass:
                mutation_candidates.append(mutation_candidate)
                mutation_resources.append(result)

        mutation_candidates = Candidates(mutation_candidates)
        mutation_candidates.update_resources(mutation_resources)

        return mutation_candidates

    def gen_crossover_candidates(self):
        """Generate specofied number of crossover candicates."""
        crossover_resources = []
        crossover_candidates: List = []
        crossover_iter = 0
        max_crossover_iters = self.num_crossover * 10
        while len(crossover_candidates) < self.num_crossover:
            crossover_iter += 1
            if crossover_iter > max_crossover_iters:
                break

            crossover_candidate = self._crossover()

            is_pass, result = self._check_constraints(
                random_subnet=crossover_candidate)
            if is_pass:
                crossover_candidates.append(crossover_candidate)
                crossover_resources.append(result)

        crossover_candidates = Candidates(crossover_candidates)
        crossover_candidates.update_resources(crossover_resources)

        return crossover_candidates

    def _mutation(self) -> SupportRandomSubnet:
        """Mutate with the specified mutate_prob."""
        candidate1 = random.choice(self.top_k_candidates.subnets)
        candidate2 = self.model.mutator.sample_choices()
        candidate = crossover(candidate1, candidate2, prob=self.mutate_prob)
        return candidate

    def _crossover(self) -> SupportRandomSubnet:
        """Crossover."""
        candidate1 = random.choice(self.top_k_candidates.subnets)
        candidate2 = random.choice(self.top_k_candidates.subnets)
        candidate = crossover(candidate1, candidate2, prob=self.crossover_prob)
        return candidate

    def _resume(self):
        """Resume searching."""
        if self.runner.rank == 0:
            searcher_resume = fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['_epoch'])
            self._max_epochs = self._max_epochs - epoch_start
            self.runner.logger.info('#' * 100)
            self.runner.logger.info(f'Resume from epoch: {epoch_start}')
            self.runner.logger.info('#' * 100)

    def _save_best_fix_subnet(self):
        """Save best subnet in searched top-k candidates."""
        if self.runner.rank == 0:
            best_random_subnet = self.top_k_candidates.subnets[0]
            self.model.mutator.set_choices(best_random_subnet)

            best_fix_subnet, sliced_model = \
                export_fix_subnet(self.model, slice_weight=True)

            timestamp_subnet = time.strftime('%Y%m%d_%H%M', time.localtime())
            model_name = f'subnet_{timestamp_subnet}.pth'
            save_path = osp.join(self.runner.work_dir, model_name)
            torch.save({
                'state_dict': sliced_model.state_dict(),
                'meta': {}
            }, save_path)
            self.runner.logger.info(f'Subnet checkpoint {model_name} saved in '
                                    f'{self.runner.work_dir}')

            save_name = 'best_fix_subnet.yaml'
            best_fix_subnet = convert_fix_subnet(best_fix_subnet)
            fileio.dump(best_fix_subnet,
                        osp.join(self.runner.work_dir, save_name))
            self.runner.logger.info(
                f'Subnet config {save_name} saved in {self.runner.work_dir}.')

            self.runner.logger.info('Search finished.')

    @torch.no_grad()
    def _val_candidate(self, use_predictor: bool = False) -> Dict:
        """Run validation.

        Args:
            use_predictor (bool): Whether to use predictor to get metrics.
                Defaults to False.
        """
        if use_predictor:
            assert self.predictor is not None
            metrics = self.predictor.predict(self.model)
        else:
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_statistics(self.runner.train_dataloader,
                                             self.calibrate_sample_num)
            self.runner.model.eval()
            for data_batch in self.dataloader:
                outputs = self.runner.model.val_step(data_batch)
                self.evaluator.process(
                    data_samples=outputs, data_batch=data_batch)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        return metrics

    def _save_searcher_ckpt(self) -> None:
        """Save searcher ckpt, which is different from common ckpt.

        It mainly contains the candicate pool, the top-k candicates with scores
        and the current epoch.
        """
        if self.runner.rank == 0:
            save_for_resume = dict()
            save_for_resume['_epoch'] = self._epoch
            for k in ['candidates', 'top_k_candidates']:
                save_for_resume[k] = getattr(self, k)
            fileio.dump(
                save_for_resume,
                osp.join(self.runner.work_dir,
                         f'search_epoch_{self._epoch}.pkl'))
            self.runner.logger.info(
                f'Epoch:[{self._epoch}/{self._max_epochs}], top1_score: '
                f'{self.top_k_candidates.scores[0]}')

            if self.max_keep_ckpts > 0:
                cur_ckpt = self._epoch + 1
                redundant_ckpts = range(1, cur_ckpt - self.max_keep_ckpts)
                for _step in redundant_ckpts:
                    ckpt_path = osp.join(self.runner.work_dir,
                                         f'search_epoch_{_step}.pkl')
                    if osp.isfile(ckpt_path):
                        os.remove(ckpt_path)

    def _check_constraints(
            self, random_subnet: SupportRandomSubnet) -> Tuple[bool, Dict]:
        """Check whether is beyond constraints.

        Returns:
            bool, result: The result of checking.
        """
        is_pass, results = check_subnet_resources(
            model=self.model,
            subnet=random_subnet,
            estimator=self.estimator,
            constraints_range=self.constraints_range)

        return is_pass, results

    def _init_predictor(self):
        """Initialize predictor, training is required."""
        if self.predictor.handler_ckpt:
            self.predictor.load_checkpoint()
            self.runner.logger.info(
                f'Loaded Checkpoints from {self.predictor.handler_ckpt}')
        else:
            self.runner.logger.info('No predictor checkpoints found. '
                                    'Start pre-training the predictor.')
            if isinstance(self.predictor.train_samples, str):
                self.runner.logger.info('Find specified samples in '
                                        f'{self.predictor.train_samples}')
                train_samples = fileio.load(self.predictor.train_samples)
                self.candidates = train_samples['subnets']
            else:
                self.runner.logger.info(
                    'Without specified samples. Start random sampling.')
                temp_num_candidates = self.num_candidates
                self.num_candidates = self.predictor.train_samples

                assert self.use_predictor is False, (
                    'Real evaluation is required when initializing predictor.')
                self.sample_candidates()
                self.update_candidates_scores()
                self.num_candidates = temp_num_candidates

            inputs = []
            for candidate in self.candidates.subnets:
                inputs.append(self.predictor.model2vector(candidate))
            inputs = np.array(inputs)
            labels = np.array(self.candidates.scores)
            self.predictor.fit(inputs, labels)
            if self.runner.rank == 0:
                predictor_dir = self.predictor.save_checkpoint(
                    osp.join(self.runner.work_dir, 'predictor'))
                self.runner.logger.info(
                    f'Predictor pre-trained, saved in {predictor_dir}.')
            self.use_predictor = True
            self.candidates = Candidates()
