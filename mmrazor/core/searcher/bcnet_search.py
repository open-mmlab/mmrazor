# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import time
import mmcv.fileio
from mmcv.runner import get_dist_info
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

from ..builder import SEARCHERS
from .evolution_search import EvolutionSearcher
from ..utils import broadcast_object_list


@SEARCHERS.register_module()
class BCNetSearcher(EvolutionSearcher):

    def __init__(self,
                 max_channel_bins,
                 min_channel_bins=1,
                 prior_init=True,
                 **kwargs):
        super(BCNetSearcher, self).__init__(**kwargs)

        self.max_channel_bins = max_channel_bins
        self.min_channel_bins = min_channel_bins
        self.prior_init = prior_init

    def _init_population(self, loss_rec):
        self.logger.info(
            f'Initializing Prior Population with {len(loss_rec)} Loss Records')
        device = next(self.algorithm.parameters()).device
        num_space_id = len(self.algorithm.pruner.channel_spaces)
        # all possible num of width value
        num_width = self.max_channel_bins - self.min_channel_bins + 1

        # Measure the potential loss ``loss_matrix`` of sampling c_l width at
        # l-th layer by recording the averaged training loss of all m widths
        # going through it. m is ``loss_rec_num``.
        loss_matrix = torch.zeros((num_space_id, num_width), device=device)
        layer_width_cnt = torch.zeros_like(loss_matrix)

        for rec in loss_rec:
            subnet_l, subnet_r, loss = rec.subnet_l, rec.subnet_r, rec.loss
            assert num_space_id == len(subnet_l)
            assert len(subnet_l) == len(subnet_r)
            for i, space_id in enumerate(sorted(subnet_l.keys())):
                out_mask = subnet_l[space_id]
                # channels to channel_bins
                width = round(out_mask.sum().item() * self.max_channel_bins / out_mask.numel())
                loss_matrix[i, width - self.min_channel_bins] += loss
                layer_width_cnt[i, width - self.min_channel_bins] += 1

        # for numerical stability, give width never chosen super large loss
        loss_matrix[layer_width_cnt == 0] = 1e-4
        loss_matrix /= (layer_width_cnt + 1e-5)

        # FLOPs calculation, layer flops is proportional to i/o channel width
        # FLOPs in different layers are different.
        Flops = torch.zeros((num_space_id, num_width, num_width), device=device)
        temp = torch.zeros((num_width, num_width), device=device)
        for row in range(num_width):
            for col in range(num_width):
                temp[row, col] = (row + 1) * (col + 1)
        temp /= self.max_channel_bins**2
        space_flops = self.algorithm.get_space_flops()
        for i, space_id in enumerate(sorted(space_flops.keys())):
            Flops[i, :, :] = space_flops[space_id] * temp

        # output sample possibility is softmax of P
        P = torch.autograd.Variable(torch.randn_like(loss_matrix), requires_grad=True)
        optim = torch.optim.SGD([P], lr=0.01)

        for _ in range(10000):
            optim.zero_grad()
            prob = F.softmax(P, 1)
            prob_shift = F.pad(prob, (0, 0, 0, 1), value=1.0 / num_width)[1:, :]
            z = (prob * loss_matrix).sum(dim=1)

            F_e = (Flops * prob.view(num_space_id, num_width, 1) * prob_shift.view(num_space_id, 1, num_width)).sum()
            loss = z.mean() + (1.0 - F_e / self.constraints['flops']) ** 2
            loss.backward()
            optim.step()
            if _ % 1000 == 0:
                self.logger.info(f'Initialize Prior Population: Epoch {_} Loss {loss.item()}')

        P.detach_()
        prob = F.softmax(P, dim=1).cpu().numpy()
        self.logger.info(f'Initialize Prior Population Done: P {prob}')

        for _ in range(self.candidate_pool_size):
            while 1:
                subnet_dict = dict()
                for i, space_id in enumerate(sorted(self.algorithm.pruner.channel_spaces.keys())):
                    out_mask = self.algorithm.pruner.channel_spaces[space_id]
                    out_channels = out_mask.size(1)
                    width = np.random.choice(a=np.arange(self.min_channel_bins, self.max_channel_bins + 1),
                                             p=prob[i])
                    new_channels = round(width / self.max_channel_bins * out_channels)
                    new_out_mask = torch.zeros_like(out_mask).bool()
                    new_out_mask[:, :new_channels] = True
                    subnet_dict[space_id] = new_out_mask
                self.algorithm.pruner.set_subnet(subnet_dict)
                if self.check_constraints():
                    self.candidate_pool.append(subnet_dict)
                    break

    def mutation(self, candidate, prob):
        mutation_subnet_dict = copy.deepcopy(candidate)
        for name, mask in candidate.items():
            if np.random.random_sample() < prob:
                mutation_subnet_dict[name] = self.get_channel_mask(mask, searching=True)
        return mutation_subnet_dict

    def crossover(self, candidate1, candidate2):
        crossover_subnet_dict = copy.deepcopy(candidate1)
        for name, mask in candidate2.items():
            if np.random.random_sample() < 0.5:
                crossover_subnet_dict[name] = mask
        return crossover_subnet_dict

    def eval_subnet(self, candidate):
        score = None
        rank = get_dist_info()[0]
        subnet_l = candidate
        self.algorithm.pruner.set_subnet(subnet_l)
        self.logger.info('eval left subnet')
        outputs = self.test_fn(self.algorithm_for_test, self.dataloader)

        if rank == 0:
            eval_result = self.dataloader.dataset.evaluate(
                outputs, self.metrics, self.metric_options)
            score_l = eval_result[self.score_key]

        subnet_r = self.algorithm.pruner.reverse_subnet(subnet_l)
        self.algorithm.pruner.set_subnet(subnet_r)
        self.logger.info('eval right subnet')
        outputs = self.test_fn(self.algorithm_for_test, self.dataloader)

        if rank == 0:
            eval_result = self.dataloader.dataset.evaluate(
                outputs, self.metrics, self.metric_options)
            score_r = eval_result[self.score_key]
            score = (score_l + score_r) / 2
            self.candidate_pool_with_score[score] = candidate

        return score

    def search(self):
        rank = get_dist_info()[0]
        if self.prior_init:
            if rank == 0:
                loss_rec = torch.load('bcnet_supernet/loss_rec_epoch_299.pth')
                self._init_population(loss_rec)
                broadcast_candidate_pool = self.candidate_pool
            else:
                broadcast_candidate_pool = [None] * self.candidate_pool_size
            broadcast_candidate_pool = broadcast_object_list(
                broadcast_candidate_pool)
            self.candidate_pool = broadcast_candidate_pool

        epoch_start = 0
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

        for epoch in range(epoch_start, self.max_epoch):
            if rank == 0:
                while len(self.candidate_pool) < self.candidate_pool_size:
                    candidate = \
                        self.algorithm.pruner.sample_subnet(searching=True)
                    self.algorithm.pruner.set_subnet(candidate)

                    if self.check_constraints():
                        self.candidate_pool.append(candidate)

                broadcast_candidate_pool = self.candidate_pool
            else:
                broadcast_candidate_pool = [None] * self.candidate_pool_size
            broadcast_candidate_pool = broadcast_object_list(
                broadcast_candidate_pool)

            for i, candidate in enumerate(broadcast_candidate_pool):
                score = self.eval_subnet(candidate)
                if rank == 0:
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
                    mutation_candidate = self.mutation(candidate, self.mutate_prob)
                    self.algorithm.pruner.set_subnet(mutation_candidate)
                    if self.check_constraints():
                        mutation_candidates.append(mutation_candidate)

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

                    crossover_candidate = self.crossover(random_candidate1, random_candidate2)
                    self.algorithm.pruner.set_subnet(crossover_candidate)
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
            self.candidate_pool = broadcast_object_list(self.candidate_pool)

        if rank == 0:
            for i, (score, subnet) in enumerate(self.top_k_candidates_with_score.items()):
                mmcv.fileio.dump(
                    subnet,
                    osp.join(self.work_dir, f'subnet_top{i}_score_{score}.yaml'))
