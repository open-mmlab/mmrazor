# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import numpy as np
from mmengine import fileio

try:
    from pymoo.algorithms.so_genetic_algorithm import GA
    from pymoo.factory import get_algorithm, get_crossover, get_mutation
    from pymoo.optimize import minimize
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError:
    from mmrazor.utils import get_placeholder
    GA = get_placeholder('pymoo')
    get_algorithm = get_placeholder('pymoo')
    get_crossover = get_placeholder('pymoo')
    get_mutation = get_placeholder('pymoo')
    minimize = get_placeholder('pymoo')
    NonDominatedSorting = get_placeholder('pymoo')

from mmrazor.registry import LOOPS
from mmrazor.structures import Candidates
from .attentive_search_loop import AttentiveSearchLoop
from .utils.pymoo_utils import (AuxiliarySingleLevelProblem,
                                HighTradeoffPoints, SubsetProblem)
from .utils.pymoo_utils.problems import (BinaryCrossover, RandomMutation,
                                         RandomSampling)


@LOOPS.register_module()
class NSGA2SearchLoop(AttentiveSearchLoop):
    """Evolution search loop with NSGA2 optimizer."""

    def run_epoch(self) -> None:
        """Iterate one epoch.

        Steps:
            0. Collect archives and predictor.
            1. Sample some new candidates from the supernet.Then Append them
                to the candidates, Thus make its number equal to the specified
                number.
            2. Validate these candidates(step 1) and update their scores.
            3. Pick the top k candidates based on the scores(step 2), which
                will be used in mutation and crossover.
            4. Implement Mutation and crossover, generate better candidates.
        """
        self.archive = Candidates()
        if len(self.candidates) > 0:
            for subnet, score, flops in zip(
                    self.candidates.subnets, self.candidates.scores,
                    self.candidates.resources('flops')):
                self.update_candidates_scores()
                if self.trade_off['max_score_key'] != 0:
                    score = self.trade_off['max_score_key'] - score
                self.archive.append(subnet)
                self.archive.set_score(-1, score)
                self.archive.set_resource(-1, flops, 'flops')

        self.sample_candidates(random=(self._epoch == 0), archive=self.archive)
        self.update_candidates_scores()

        scores_before = self.top_k_candidates.scores
        self.runner.logger.info(f'top k scores before update: '
                                f'{scores_before}')

        self.candidates.extend(self.top_k_candidates)
        self.sort_candidates()
        self.top_k_candidates = \
            Candidates(self.candidates[:self.top_k])  # type: ignore

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

    def sample_candidates(self, random: bool = True, archive=None) -> None:
        if random:
            super().sample_candidates()
        else:
            candidates = self.sample_candidates_with_nsga2(
                archive, self.num_candidates)
            new_candidates = []
            candidates_resources = []
            for candidate in candidates:
                is_pass, result = self._check_constraints(candidate)
                if is_pass:
                    new_candidates.append(candidate)
                    candidates_resources.append(result)
            self.candidates = Candidates(new_candidates)

            if len(candidates_resources) > 0:
                self.candidates.update_resources(
                    candidates_resources,
                    start=len(self.candidates) - len(candidates_resources))

    def sample_candidates_with_nsga2(self, archive: Candidates,
                                     num_candidates):
        """Searching for candidates with high-fidelity evaluation."""
        F = np.column_stack((archive.scores, archive.resources('flops')))
        front_index = NonDominatedSorting().do(
            F, only_non_dominated_front=True)

        fronts = np.array(archive.subnets)[front_index]
        fronts = np.array(
            [self.predictor.model2vector(cand) for cand in fronts])
        fronts = self.predictor.preprocess(fronts)

        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(self, len(fronts[0]))

        # initiate a multi-objective solver to optimize the problem
        method = get_algorithm(
            'nsga2',
            pop_size=40,
            sampling=fronts,
            crossover=get_crossover('int_two_point', prob=0.9),
            mutation=get_mutation('int_pm', eta=1.0),
            eliminate_duplicates=True)

        result = minimize(problem, method, ('n_gen', 4), seed=1, verbose=True)

        check_list = []
        for x in result.pop.get('X'):
            check_list.append(self.predictor.vector2model(x))

        not_duplicate = np.logical_not(
            [x in archive.subnets for x in check_list])

        sub_problem = SubsetProblem(result.pop[not_duplicate].get('F')[:, 1],
                                    F[front_index, 1], num_candidates)

        sub_method = GA(
            pop_size=100,
            sampling=RandomSampling(),
            crossover=BinaryCrossover(),
            mutation=RandomMutation(),
            eliminate_duplicates=True)

        sub_result = minimize(
            sub_problem, sub_method, ('n_gen', 2), seed=1, verbose=False)
        indices = sub_result.X

        _X = result.pop.get('X')[not_duplicate][indices]

        candidates = []
        for x in _X:
            x = x[0] if isinstance(x[0], list) else x
            candidates.append(self.predictor.vector2model(x))

        return candidates

    def sort_candidates(self) -> None:
        """Support sort candidates in single and multiple-obj optimization."""
        assert self.trade_off is not None, (
            '`self.trade_off` is required when sorting candidates in '
            'NSGA2SearchLoop. Got `self.trade_off` is None.')

        max_score_key = self.trade_off.get('max_score_key', 100)
        max_score_key = np.array(max_score_key)

        patches = []
        for score, flops in zip(self.candidates.scores,
                                self.candidates.resources('flops')):
            patches.append((score, flops))
        patches = np.array(patches)

        if max_score_key != 0:
            patches[:, 0] = max_score_key - patches[:, 0]  # type: ignore

        sort_idx = np.argsort(patches[:, 0])  # type: ignore
        F = patches[sort_idx]

        dm = HighTradeoffPoints(n_survive=len(patches))
        candidate_index = dm.do(F)
        candidate_index = sort_idx[candidate_index]

        self.candidates = \
            [self.candidates[idx] for idx in candidate_index]  # type: ignore

    def _save_searcher_ckpt(self):
        """Save searcher ckpt, which is different from common ckpt.

        It mainly contains the candicate pool, the top-k candicates with scores
        and the current epoch.
        """
        if self.runner.rank == 0:
            rmse, rho, tau = 0, 0, 0
            if len(self.archive) > 0:
                top1_err_pred = self.fit_predictor(self.archive)

                self.candidates = self.archive
                self.use_predictor = False
                self.update_candidates_scores()

                rmse, rho, tau = self.predictor.get_correlation(
                    top1_err_pred, np.array(self.candidates.scores))

            save_for_resume = dict()
            save_for_resume['_epoch'] = self._epoch
            for k in ['candidates', 'top_k_candidates']:
                save_for_resume[k] = getattr(self, k)
            fileio.dump(
                save_for_resume,
                osp.join(self.runner.work_dir,
                         f'search_epoch_{self._epoch}.pkl'))

            correlation_str = 'fitting '
            correlation_str += f'RMSE = {rmse:.4f}, '
            correlation_str += f'Spearmans Rho = {rho:.4f}, '
            correlation_str += f'Kendalls Tau = {tau:.4f}'

            self.pareto_mode = False
            if self.pareto_mode:
                step_str = '\n'
                for step, candidates in self.pareto_candidates.items():
                    if len(candidates) > 0:
                        step_str += f'step: {step}: '
                        step_str += f'{candidates[0][self.score_key]}\n'
                self.runner.logger.info(
                    f'Epoch:[{self._epoch}/{self._max_epochs}] '
                    f'Top1_score: {step_str} '
                    f'{correlation_str}')
            else:
                self.runner.logger.info(
                    f'Epoch:[{self._epoch}/{self._max_epochs}] '
                    f'Top1_score: {self.top_k_candidates.scores[0]} '
                    f'{correlation_str}')

            if self.max_keep_ckpts > 0:
                cur_ckpt = self._epoch + 1
                redundant_ckpts = range(1, cur_ckpt - self.max_keep_ckpts)
                for _step in redundant_ckpts:
                    ckpt_path = osp.join(self.runner.work_dir,
                                         f'search_epoch_{_step}.pkl')
                    if osp.isfile(ckpt_path):
                        os.remove(ckpt_path)

    def fit_predictor(self, candidates):
        """Predict performance using predictor."""
        assert self.predictor.initialize is True

        metrics = []
        for i, candidate in enumerate(candidates.subnets):
            self.model.mutator.set_choices(candidate)
            self.finetune_step(self.model)
            metric = self._val_candidate(use_predictor=True)
            metrics.append(metric[self.score_key])

        max_score_key = self.trade_off.get('max_score_key', 0.)
        assert max_score_key > 0.
        metrics = max_score_key - np.array(metrics)
        return metrics
