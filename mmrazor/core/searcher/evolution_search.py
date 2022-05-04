# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import time

import mmcv.fileio
from mmcv.runner import get_dist_info

from ..builder import SEARCHERS
from ..utils import broadcast_object_list


@SEARCHERS.register_module()
class EvolutionSearcher():
    """Implement of evolution search.

    Args:
        algorithm (:obj:`torch.nn.Module`): Algorithm to be used.
        dataloader (nn.Dataloader): Pytorch data loader.
        test_fn (function): Test api to used for evaluation.
        work_dir (str): Working direction is to save search result and log.
        logger (logging.Logger): To log info in search stage.
        candidate_pool_size (int): The length of candidate pool.
        candidate_top_k (int): Specify top k candidates based on scores.
        constraints (dict): Constraints to be used for screening candidates.
        metrics (str): Metrics to be used for evaluating candidates.
        metric_options (str): Options to be used for metrics.
        score_key (str): To be used for specifying one metric from evaluation
            results.
        max_epoch (int): Specify max epoch to end evolution search.
        num_mutation (int): The number of candidates got by mutation.
        num_crossover (int): The number of candidates got by crossover.
        mutate_prob (float): The probability of mutation.
        resume_from (str): Specify the path of saved .pkl file for resuming
            searching
    """

    def __init__(self,
                 algorithm,
                 dataloader,
                 test_fn,
                 work_dir,
                 logger,
                 candidate_pool_size=50,
                 candidate_top_k=10,
                 constraints=dict(flops=330 * 1e6),
                 metrics=None,
                 metric_options=None,
                 score_key='accuracy_top-1',
                 max_epoch=20,
                 num_mutation=25,
                 num_crossover=25,
                 mutate_prob=0.1,
                 resume_from=None,
                 **search_kwargs):

        if not hasattr(algorithm, 'module'):
            raise NotImplementedError('Do not support searching with cpu.')
        self.algorithm = algorithm.module
        self.algorithm_for_test = algorithm
        self.dataloader = dataloader
        self.constraints = constraints
        self.metrics = metrics
        self.metric_options = metric_options
        self.score_key = score_key
        self.candidate_pool = list()
        self.candidate_pool_size = candidate_pool_size
        self.max_epoch = max_epoch
        self.test_fn = test_fn
        self.candidate_top_k = candidate_top_k
        self.num_mutation = num_mutation
        self.num_crossover = num_crossover
        self.mutate_prob = mutate_prob
        self.top_k_candidates_with_score = dict()
        self.candidate_pool_with_score = dict()
        self.work_dir = work_dir
        self.resume_from = resume_from
        self.logger = logger

    def check_constraints(self):
        """Check whether is beyond constraints.

        Returns:
            bool: The result of checking.
        """
        flops = self.algorithm.get_subnet_flops()
        if flops < self.constraints['flops']:
            return True
        else:
            return False

    def update_top_k(self):
        """Update top k candidates."""
        self.top_k_candidates_with_score.update(self.candidate_pool_with_score)
        self.top_k_candidates_with_score = dict(
            sorted(
                self.top_k_candidates_with_score.items(),
                key=lambda x: x[0],
                reverse=True))
        keys = list(self.top_k_candidates_with_score.keys())
        new_dict = dict()
        for k in keys[:self.candidate_top_k]:
            new_dict[k] = self.top_k_candidates_with_score[k]
        self.top_k_candidates_with_score = new_dict.copy()

    def search(self):
        """Execute the pipeline of evolution search."""
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        rank = get_dist_info()[0]
        for epoch in range(epoch_start, self.max_epoch):
            if rank == 0:
                while len(self.candidate_pool) < self.candidate_pool_size:
                    candidate = \
                        self.algorithm.mutator.sample_subnet(searching=True)
                    self.algorithm.mutator.set_subnet(candidate)

                    if self.check_constraints():
                        self.candidate_pool.append(candidate)
            else:
                self.candidate_pool = [None] * self.candidate_pool_size
            broadcast_object_list(self.candidate_pool)

            for i, candidate in enumerate(self.candidate_pool):
                self.algorithm.mutator.set_subnet(candidate)
                outputs = self.test_fn(self.algorithm_for_test,
                                       self.dataloader)

                if rank == 0:
                    eval_result = self.dataloader.dataset.evaluate(
                        outputs, self.metrics, self.metric_options)
                    score = eval_result[self.score_key]
                    self.candidate_pool_with_score[score] = candidate
                    self.logger.info(f'Epoch:[{epoch + 1}/{self.max_epoch}] '
                                     f'Candidate:[{i + 1}/'
                                     f'{self.candidate_pool_size}] '
                                     f'Score:{score}')
            if rank == 0:
                scores_before = list(self.top_k_candidates_with_score.keys())
                self.logger.info(f'top k scores before update: '
                                 f'{scores_before}')
                self.update_top_k()
                scores_after = list(self.top_k_candidates_with_score.keys())
                self.logger.info(f'top k scores before update: '
                                 f'{scores_after}')

                mutation_candidates = list()
                max_mutate_iters = self.num_mutation * 10
                mutate_iter = 0
                while len(mutation_candidates) < self.num_mutation:
                    mutate_iter += 1
                    if mutate_iter > max_mutate_iters:
                        break
                    candidate = random.choice(
                        list(self.top_k_candidates_with_score.values()))
                    mutation = self.algorithm.mutator.mutation(
                        candidate, self.mutate_prob)
                    self.algorithm.mutator.set_subnet(mutation)
                    if self.check_constraints():
                        mutation_candidates.append(mutation)

                crossover_candidates = list()
                crossover_iter = 0
                max_crossover_iters = self.num_crossover * 10
                while len(crossover_candidates) < self.num_crossover:
                    crossover_iter += 1
                    if crossover_iter > max_crossover_iters:
                        break

                    random_candidate1 = random.choice(
                        list(self.top_k_candidates_with_score.values()))
                    random_candidate2 = random.choice(
                        list(self.top_k_candidates_with_score.values()))

                    crossover_candidate = \
                        self.algorithm.mutator.crossover(
                            random_candidate1, random_candidate2)
                    self.algorithm.mutator.set_subnet(crossover_candidate)
                    if self.check_constraints():
                        crossover_candidates.append(crossover_candidate)

                self.candidate_pool = (
                    mutation_candidates + crossover_candidates)

                save_for_resume = dict()
                save_for_resume['epoch'] = epoch + 1
                for k in ['candidate_pool', 'top_k_candidates_with_score']:
                    save_for_resume[k] = getattr(self, k)
                mmcv.fileio.dump(
                    save_for_resume,
                    osp.join(self.work_dir, f'search_epoch_{epoch + 1}.pkl'))
                self.logger.info(
                    f'Epoch:[{epoch + 1}/{self.max_epoch}], top1_score: '
                    f'{list(self.top_k_candidates_with_score.keys())[0]}')
            broadcast_object_list(self.candidate_pool)

        if rank == 0:
            final_subnet_dict = list(
                self.top_k_candidates_with_score.values())[0]
            self.algorithm.mutator.set_chosen_subnet(final_subnet_dict)
            final_subnet_dict_to_save = dict()
            for k in final_subnet_dict.keys():
                final_subnet_dict_to_save[k] = dict({
                    'chosen':
                    self.algorithm.mutator.search_spaces[k]['chosen']
                })
            timestamp_subnet = time.strftime('%Y%m%d_%H%M', time.localtime())
            save_name = f'final_subnet_{timestamp_subnet}.yaml'
            mmcv.fileio.dump(final_subnet_dict_to_save,
                             osp.join(self.work_dir, save_name))
            self.logger.info('Search finished and '
                             f'{save_name} saved in {self.work_dir}.')
