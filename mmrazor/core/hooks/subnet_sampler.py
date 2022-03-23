# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import random
from abc import abstractmethod

import mmcv
import numpy as np
import torch.distributed as dist
from mmcls.models.losses import accuracy
from mmcv.runner import HOOKS, Hook
from torch.utils.data import DataLoader


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@HOOKS.register_module()
class BaseSamplerHook(Hook):

    def __init__(self):
        super(BaseSamplerHook, self).__init__()
        pass

    @abstractmethod
    def _do_sample(self, runner):
        pass


@HOOKS.register_module()
class GreedySamplerHook(BaseSamplerHook):

    def __init__(self,
                 dataloader,
                 log_interval=50,
                 score_key='accuracy_top-1',
                 constraints=dict(flops=330 * 1e6),
                 pool_size=1000,
                 sample_num=10,
                 top_k=5,
                 p_stg='linear',
                 start_iter=10000,
                 max_iter=144360,
                 init_p=0.,
                 max_p=0.8,
                 **eval_kwargs):
        super(GreedySamplerHook, self).__init__()

        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')

        self.log_interval = log_interval
        self.score_key = score_key

        self.constraints = constraints
        self.pool_size = pool_size
        self.candidate_pool = []

        self.sample_num = sample_num
        self.top_k = top_k
        self.top_k_subnets = []

        assert p_stg in ['linear', 'consine']
        self.p_stg = p_stg
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.init_p = init_p
        self.max_p = max_p
        self.cur_p = 0.

        self.eval_batch_data = next(IterLoader(dataloader))
        self.eval_kwargs = eval_kwargs

    def _update_candidate_pool(self):
        self.candidate_pool.sort(key=lambda x: x[1], reverse=True)
        self.candidate_pool = self.candidate_pool[:self.pool_size]

    def _evaluate_batch(self, results, gt_labels):
        topk = (1, 5)
        results = np.vstack(results)
        gt_labels = np.expand_dims(np.array(gt_labels), axis=0)
        acc = accuracy(results, gt_labels, topk=topk)
        eval_results_ = {f'accuracy_top-{k}': a for k, a in zip(topk, acc)}
        return eval_results_

    def _gen_subnet_from_supernet(self, model):
        subnet_dict = model.mutator.sample_subnet(searching=True)
        return subnet_dict

    def _gen_subnet_from_pool(self):
        assert len(self.candidate_pool) > 0
        subnet_dict = random.choice(self.candidate_pool)[0]
        return subnet_dict

    def before_train_iter(self, runner):
        if runner.iter >= self.start_iter:
            if runner.rank == 0:
                runner.model.module.subnet = self._do_sample(runner)
            for k in runner.model.module.subnet.keys():
                dist.broadcast(runner.model.module.subnet[k], src=0)

    def after_train_iter(self, runner):
        if runner.iter >= runner.max_iters - 1:
            runner.logger.info(
                f'cur_iter: {runner.iter}; max_iters: {runner.max_iters}')
            if runner.rank == 0:
                self._save_candidate_pool(runner)

    def _save_candidate_pool(self, runner):
        save_path = os.path.join(runner.work_dir, 'candidate_pool.pkl')
        mmcv.fileio.dump(self.candidate_pool, save_path)
        runner.logger.info(f'candidate_pool.pkl saved in {runner.work_dir}')

    def _do_sample(self, runner):
        if not hasattr(runner.model, 'module'):
            raise NotImplementedError('Do not support '
                                      'GreedySamplerHook with cpu.')
        if len(self.top_k_subnets) == 0:
            cur_iter = runner.iter
            if cur_iter > self.max_iter:
                p = self.max_p
            elif cur_iter < self.start_iter:
                p = self.init_p
            elif self.p_stg == 'linear':
                p = (self.max_p - self.init_p) / \
                    (self.max_iter - self.start_iter) * \
                    (cur_iter - self.start_iter)
            elif self.p_stg == 'consine':
                p = (1 - self.init_p) * 0.5 * \
                    (1 + math.cos(math.pi *
                                  (cur_iter - self.start_iter) /
                                  (self.max_iter - self.start_iter))) \
                    + self.init_p
            self.cur_p = p

            eval_subnets = []
            uniform_num = 0

            for _ in range(self.sample_num):
                if random.random() >= self.cur_p or \
                        len(self.candidate_pool) == 0:
                    eval_subnets.append(
                        self._gen_subnet_from_supernet(runner.model.module))
                    uniform_num += 1
                else:
                    eval_subnets.append(self._gen_subnet_from_pool())

            eval_results = []
            for subnet in eval_subnets:
                runner.model.module.mutator.set_subnet(subnet)
                results = runner.model(
                    self.eval_batch_data['img'], return_loss=False)
                eval_res = self._evaluate_batch(
                    results, self.eval_batch_data['gt_label'])
                score = eval_res[self.score_key]
                eval_results.append([subnet, score])
                if len(self.constraints) == 0:
                    self.candidate_pool.append([subnet, score])
                else:
                    flops = runner.model.module.get_subnet_flops()
                    if flops <= self.constraints['flops']:
                        self.candidate_pool.append([subnet, score])
            self._update_candidate_pool()

            eval_results.sort(key=lambda x: x[1], reverse=True)
            for x in eval_results[:self.top_k]:
                self.top_k_subnets.append(x[0])
            top1_score = eval_results[0][1].item()

            if cur_iter % self.log_interval < self.top_k:
                runner.logger.info(f'GreedySampler: [{cur_iter:>6d}] '
                                   f'prob {self.cur_p:.3f} '
                                   f'uniform_num '
                                   f'{uniform_num}/{self.sample_num} '
                                   f'top1_score {top1_score:.3f} '
                                   f'candidate_pool: '
                                   f'{len(self.candidate_pool)}')
        return self.top_k_subnets.pop(0)
