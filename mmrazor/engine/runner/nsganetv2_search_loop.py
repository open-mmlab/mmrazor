# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
from mmengine import fileio
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mmrazor.models.task_modules import (AuxiliarySingleLevelProblem,
                                         GeneticOptimizer, NSGA2Optimizer,
                                         SubsetProblem)
from mmrazor.registry import LOOPS
from mmrazor.structures import Candidates, export_fix_subnet
from .attentive_search_loop import AttentiveSearchLoop
from .utils.high_tradeoff_points import HighTradeoffPoints

# from pymoo.algorithms.moo.nsga2 import NSGA2 as NSGA2Optimizer
# from pymoo.algorithms.soo.nonconvex.ga import GA as GeneticOptimizer
# from pymoo.optimize import minimize


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
        archive = Candidates()
        if len(self.candidates) > 0:
            for subnet, score, flops in zip(
                    self.candidates.subnets, self.candidates.scores,
                    self.candidates.resources('flops')):
                if self.trade_off['max_score_key'] != 0:
                    score = self.trade_off['max_score_key'] - score
                archive.append(subnet)
                archive.set_score(-1, score)
                archive.set_resource(-1, flops, 'flops')

        self.sample_candidates(random=(self._epoch == 0), archive=archive)
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
        method = NSGA2Optimizer(
            pop_size=40,
            sampling=fronts,  # initialize with current nd archs
            eliminate_duplicates=True,
            logger=self.runner.logger)

        # # kick-off the search
        method.initialize(problem, n_gen=20, verbose=True)
        result = method.solve()

        # check for duplicates
        check_list = []
        for x in result['pop'].get('X'):
            check_list.append(self.predictor.vector2model(x))

        not_duplicate = np.logical_not(
            [x in archive.subnets for x in check_list])

        # extra process after nsga2 search
        sub_problem = SubsetProblem(
            result['pop'][not_duplicate].get('F')[:, 1], F[front_index, 1],
            num_candidates)
        sub_method = GeneticOptimizer(
            pop_size=num_candidates, eliminate_duplicates=True)
        sub_method.initialize(sub_problem, n_gen=4, verbose=False)
        indices = sub_method.solve()['X']

        candidates = []
        pop = result['pop'][not_duplicate][indices]
        for x in pop.get('X'):
            x = x[0] if isinstance(x[0], list) else x
            candidates.append(self.predictor.vector2model(x))

        return candidates

    def sort_candidates(self) -> None:
        """Support sort candidates in single and multiple-obj optimization."""
        assert self.trade_off is not None, (
            '`self.trade_off` is required when sorting candidates in '
            'NSGA2SearchLoop. Got `self.trade_off` is None.')
        ratio = self.trade_off.get('ratio', 1)
        max_score_key = self.trade_off.get('max_score_key', 100)
        max_score_key = np.array(max_score_key)

        multi_obj_score = []
        for score, flops in zip(self.candidates.scores,
                                self.candidates.resources('flops')):
            multi_obj_score.append((score, flops))
        multi_obj_score = np.array(multi_obj_score)

        if max_score_key != 0:
            multi_obj_score[:, 0] = max_score_key - multi_obj_score[:, 0]

        sort_idx = np.argsort(multi_obj_score[:, 0])
        F = multi_obj_score[sort_idx]

        dm = HighTradeoffPoints(ratio, n_survive=len(multi_obj_score))
        candidate_index = dm.do(F)
        candidate_index = sort_idx[candidate_index]

        self.candidates = [self.candidates[idx] for idx in candidate_index]

    def _save_searcher_ckpt(self, archive=[]):
        """Save searcher ckpt, which is different from common ckpt.

        It mainly contains the candicate pool, the top-k candicates with scores
        and the current epoch.
        """
        if self.runner.rank == 0:
            rmse, rho, tau = 0, 0, 0
            if len(archive) > 0:
                top1_err_pred = self.fit_predictor(archive)
                rmse, rho, tau = self.predictor.get_correlation(
                    top1_err_pred, np.array([x[1] for x in archive]))

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
                    f'Epoch:[{self._epoch + 1}/{self._max_epochs}], '
                    f'top1_score: {step_str} '
                    f'{correlation_str}')
            else:
                self.runner.logger.info(
                    f'Epoch:[{self._epoch + 1}/{self._max_epochs}], '
                    f'top1_score: {self.top_k_candidates.scores[0]} '
                    f'{correlation_str}')

    def fit_predictor(self, candidates):
        """anticipate testfn training(err rate)."""
        inputs = [export_fix_subnet(x) for x in candidates.subnets]
        inputs = np.array([self.predictor.model2vector(x) for x in inputs])

        targets = np.array([x[1] for x in candidates])

        if not self.predictor.pretrained:
            self.predictor.fit(inputs, targets)

        metrics = self.predictor.predict(inputs)
        if self.max_score_key != 0:
            for i in range(len(metrics)):
                metrics[i] = self.max_score_key - metrics[i]
        return metrics

    def finetune_step(self, model):
        """Fintune before candidates evaluation."""
        self.runner.logger.info('Start finetuning...')
        self.finetune_runner.model = model
        self.finetune_runner.call_hook('before_run')

        self.finetune_runner.optim_wrapper.initialize_count_status(
            self.finetune_runner.model, self.finetune_runner._train_loop.iter,
            self.finetune_runner._train_loop.max_iters)

        self.model = self.finetune_runner.train_loop.run()
        self.finetune_runner.train_loop._iter = 0

        self.finetune_runner.call_hook('after_run')
        self.runner.logger.info('End finetuning...')
