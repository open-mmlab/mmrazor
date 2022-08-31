# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine import fileio
from mmengine.dist import broadcast_object_list
from mmengine.evaluator import Evaluator
from mmengine.runner import EpochBasedTrainLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmrazor.models.task_modules.estimators import get_model_complexity_info
from mmrazor.registry import LOOPS
from mmrazor.structures import Candidates, export_fix_subnet, load_fix_subnet
from mmrazor.utils import SupportRandomSubnet
from .utils import crossover


@LOOPS.register_module()
class EvolutionSearchLoop(EpochBasedTrainLoop):
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
        flops_range (tuple, optional): flops_range to be used for screening
            candidates.
        spec_modules (list): Used for specify modules need to counter.
            Defaults to list().
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
                 flops_range: Optional[Tuple[float, float]] = (0., 330 * 1e6),
                 spec_modules: List = [],
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
        self.flops_range = flops_range
        self.spec_modules = spec_modules
        self.score_key = score_key
        self.num_mutation = num_mutation
        self.num_crossover = num_crossover
        self.mutate_prob = mutate_prob
        self.max_keep_ckpts = max_keep_ckpts
        self.resume_from = resume_from

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

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')

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
            1. Sample some new candidates from the supernet.Then Append them
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
        self.candidates.sort(key=lambda x: x[1], reverse=True)
        self.top_k_candidates = Candidates(self.candidates[:self.top_k])

        scores_after = self.top_k_candidates.scores
        self.runner.logger.info(f'top k scores after update: '
                                f'{scores_after}')

        mutation_candidates = self.gen_mutation_candidates()
        crossover_candidates = self.gen_crossover_candidates()
        candidates = mutation_candidates + crossover_candidates
        assert len(candidates) <= self.num_candidates, 'Total of mutation and \
            crossover should be no more than the number of candidates.'

        self.candidates = Candidates(candidates)
        self._epoch += 1

    def sample_candidates(self) -> None:
        """Update candidate pool contains specified number of candicates."""
        if self.runner.rank == 0:
            while len(self.candidates) < self.num_candidates:
                candidate = self.model.sample_subnet()
                if self._check_constraints(random_subnet=candidate):
                    self.candidates.append(candidate)
        else:
            self.candidates = Candidates([None] * self.num_candidates)
        # broadcast candidates to val with multi-GPUs.
        broadcast_object_list(self.candidates.data)

    def update_candidates_scores(self) -> None:
        """Validate candicate one by one from the candicate pool, and update
        top-k candicates."""
        for i, candidate in enumerate(self.candidates.subnets):
            self.model.set_subnet(candidate)
            metrics = self._val_candidate()
            score = metrics[self.score_key] \
                if len(metrics) != 0 else 0.
            self.candidates.set_score(i, score)
            self.runner.logger.info(
                f'Epoch:[{self._epoch}/{self._max_epochs}] '
                f'Candidate:[{i + 1}/{self.num_candidates}] '
                f'Score:{score}')

    def gen_mutation_candidates(self) -> List:
        """Generate specified number of mutation candicates."""
        mutation_candidates: List = []
        max_mutate_iters = self.num_mutation * 10
        mutate_iter = 0
        while len(mutation_candidates) < self.num_mutation:
            mutate_iter += 1
            if mutate_iter > max_mutate_iters:
                break

            mutation_candidate = self._mutation()

            if self._check_constraints(random_subnet=mutation_candidate):
                mutation_candidates.append(mutation_candidate)
        return mutation_candidates

    def gen_crossover_candidates(self) -> List:
        """Generate specofied number of crossover candicates."""
        crossover_candidates: List = []
        crossover_iter = 0
        max_crossover_iters = self.num_crossover * 10
        while len(crossover_candidates) < self.num_crossover:
            crossover_iter += 1
            if crossover_iter > max_crossover_iters:
                break

            crossover_candidate = self._crossover()

            if self._check_constraints(random_subnet=crossover_candidate):
                crossover_candidates.append(crossover_candidate)
        return crossover_candidates

    def _mutation(self) -> SupportRandomSubnet:
        """Mutate with the specified mutate_prob."""
        candidate1 = random.choice(self.top_k_candidates.subnets)
        candidate2 = self.model.sample_subnet()
        candidate = crossover(candidate1, candidate2, prob=self.mutate_prob)
        return candidate

    def _crossover(self) -> SupportRandomSubnet:
        """Crossover."""
        candidate1 = random.choice(self.top_k_candidates.subnets)
        candidate2 = random.choice(self.top_k_candidates.subnets)
        candidate = crossover(candidate1, candidate2)
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
            self.model.set_subnet(best_random_subnet)
            best_fix_subnet = export_fix_subnet(self.model)
            save_name = 'best_fix_subnet.yaml'
            fileio.dump(best_fix_subnet,
                        osp.join(self.runner.work_dir, save_name))
            self.runner.logger.info(
                'Search finished and '
                f'{save_name} saved in {self.runner.work_dir}.')

    @torch.no_grad()
    def _val_candidate(self) -> Dict:
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

    def _check_constraints(self, random_subnet: SupportRandomSubnet) -> bool:
        """Check whether is beyond constraints.

        Returns:
            bool: The result of checking.
        """
        if self.flops_range is None:
            return True

        self.model.set_subnet(random_subnet)
        fix_mutable = export_fix_subnet(self.model)
        copied_model = copy.deepcopy(self.model)
        load_fix_subnet(copied_model, fix_mutable)
        flops, _ = get_model_complexity_info(
            copied_model, spec_modules=self.spec_modules)

        if self.flops_range[0] <= flops <= self.flops_range[1]:
            return True
        else:
            return False
