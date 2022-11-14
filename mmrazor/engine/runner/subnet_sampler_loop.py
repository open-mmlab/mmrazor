# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine import fileio
from mmengine.evaluator import Evaluator
from mmengine.runner import IterBasedTrainLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS, TASK_UTILS
from mmrazor.structures import Candidates
from mmrazor.utils import SupportRandomSubnet
from .utils import check_subnet_resources


class BaseSamplerTrainLoop(IterBasedTrainLoop):
    """IterBasedTrainLoop for base sampler.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader for training the model.
        max_iters (int): Total training iters.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 max_iters: int,
                 val_begin: int = 1,
                 val_interval: int = 1000):
        super().__init__(runner, dataloader, max_iters, val_begin,
                         val_interval)
        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model

    @abstractmethod
    def sample_subnet(self) -> SupportRandomSubnet:
        """Sample a subnet to train the supernet."""

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        subnet = self.sample_subnet()
        self.model.set_subnet(subnet)
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.message_hub.update_info('train_logs', outputs)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


@LOOPS.register_module()
class GreedySamplerTrainLoop(BaseSamplerTrainLoop):
    """IterBasedTrainLoop for greedy sampler.
    In GreedySamplerTrainLoop, `Greedy` means that only use some top
    sampled candidates to train the supernet. So GreedySamplerTrainLoop mainly
    picks the top candidates based on their val socres, then use them to train
    the supernet one by one.
    Steps:
        1. Sample from the supernet and the candidates.
        2. Validate these sampled candidates to get each candidate's score.
        3. Get top-k candidates based on their scores, then use them to train
        the supernet one by one.
    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader for training the model.
        dataloader_val (Dataloader or dict): A dataloader object or a dict to
            build a dataloader for evaluating the candidates.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        max_iters (int): Total training iters.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        score_key (str): Specify one metric in evaluation results to score
            candidates. Defaults to 'accuracy_top-1'.
        constraints_range (Dict[str, Any]): Constraints to be used for
            screening candidates. Defaults to dict(flops=(0, 330)).
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to None.
        num_candidates (int): The number of the candidates consist of samples
            from supernet and itself. Defaults to 1000.
        num_samples (int): The number of sample in each sampling subnet.
            Defaults to 10.
        top_k (int): Choose top_k subnet from the candidates used to train
            the supernet. Defaults to 5.
        prob_schedule (str): The schedule to generate the probablity of
            sampling from the candidates. The probablity will increase from
            [init_prob, max_prob] during [schedule_start_iter,
            schedule_end_iter]. Both of 'linear' schedule and 'consine'
            schedule are supported. Defaults to 'linear'.
        schedule_start_iter (int): The start iter of the prob_schedule.
            Defaults to 10000. 10000 is corresponding to batch_size: 1024.
            You should adptive it based on your batch_size.
        schedule_end_iter (int): The end iter in of the prob_schedule.
            Defaults to 144360. 144360 = 120(epoch) * 1203 (iters/epoch),
            batch_size is 1024. You should adptive it based on the batch_size
            and the total training epochs.
        init_prob (float): The init probablity of the prob_schedule.
            Defaults to 0.0.
        max_prob (float): The max probablity of the prob_schedule.
            Defaults to 0.8.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 dataloader_val: Union[Dict, DataLoader],
                 evaluator: Union[Evaluator, Dict, List],
                 max_iters: int,
                 val_begin: int = 1,
                 val_interval: int = 1000,
                 score_key: str = 'accuracy/top1',
                 constraints_range: Dict[str, Any] = dict(flops=(0, 330)),
                 estimator_cfg: Optional[Dict] = None,
                 num_candidates: int = 1000,
                 num_samples: int = 10,
                 top_k: int = 5,
                 prob_schedule: str = 'linear',
                 schedule_start_iter: int = 10000,
                 schedule_end_iter: int = 144360,
                 init_prob: float = 0.,
                 max_prob: float = 0.8) -> None:
        super().__init__(runner, dataloader, max_iters, val_begin,
                         val_interval)
        if isinstance(dataloader_val, dict):
            self.dataloader_val = runner.build_dataloader(
                dataloader_val, seed=runner.seed)
        else:
            self.dataloader_val = dataloader_val

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)
        else:
            self.evaluator = evaluator

        self.score_key = score_key
        self.constraints_range = constraints_range
        self.num_candidates = num_candidates
        self.num_samples = num_samples
        self.top_k = top_k
        assert prob_schedule in ['linear', 'consine']
        self.prob_schedule = prob_schedule
        self.schedule_start_iter = schedule_start_iter
        self.schedule_end_iter = schedule_end_iter
        self.init_prob = init_prob
        self.max_prob = max_prob
        self.cur_prob: float = 0.

        self.candidates = Candidates()
        self.top_k_candidates = Candidates()

        # initialize estimator
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            if (self.runner.val_loop is not None
                    and self._iter >= self.runner.val_begin
                    and self._iter % self.runner.val_interval == 0):
                self.runner.val_loop.run()
        self._save_candidates()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')

    def sample_subnet(self) -> SupportRandomSubnet:
        """Sample a subnet from top_k candidates one by one, then to train the
        surpernet with the subnet.

        Steps:
            1. Update and get the `top_k_candidates`.
                1.1. Update the prob of sampling from the `candidates` based on
                the `prob_schedule` and the current iter.
                1.2. Sample `num_samples` candidates from the supernet and the
                `candidates` based on the updated prob(step 1.1).
                1.3. Val all candidates to get their scores, including the
                sampled candidates(step 1.2).
                1.4. Update the `top_k_candidates` based on
                their scores(step 1.3).
            2. Pop from the `top_k_candidates` one by one to train
            the supernet.
        """
        if len(self.top_k_candidates) == 0:
            self.update_cur_prob(cur_iter=self._iter)

            sampled_candidates, num_sample_from_supernet = \
                self.get_candidates_with_sample(num_samples=self.num_samples)

            self.candidates.extend(sampled_candidates)

            self.update_candidates_scores()

            self.candidates.sort_by(key_indicator='score', reverse=True)
            self.candidates = Candidates(
                self.candidates.data[:self.num_candidates])
            self.top_k_candidates = Candidates(
                self.candidates.data[:self.top_k])

            top1_score = self.top_k_candidates.scores[0]
            if (self._iter % self.val_interval) < self.top_k:
                self.runner.logger.info(
                    f'GreedySampler: [{self._iter:>6d}] '
                    f'prob {self.cur_prob:.3f} '
                    f'num_sample_from_supernet '
                    f'{num_sample_from_supernet}/{self.num_samples} '
                    f'top1_score {top1_score:.3f} '
                    f'cur_num_candidates: {len(self.candidates)}')
        return self.top_k_candidates.subnets[0]

    def update_cur_prob(self, cur_iter: int) -> None:
        """update current probablity of sampling from the candidates, which is
        generated based on the probablity strategy and current iter."""
        if cur_iter > self.schedule_end_iter:
            self.cur_prob = self.max_prob
        elif cur_iter < self.schedule_start_iter:
            self.cur_prob = self.init_prob
        else:
            schedule_all_steps = self.schedule_end_iter - \
                self.schedule_start_iter
            schedule_cur_steps = cur_iter - self.schedule_start_iter
            if self.prob_schedule == 'linear':
                tmp = self.max_prob - self.init_prob
                self.cur_prob = tmp / schedule_all_steps * schedule_cur_steps
            elif self.prob_schedule == 'consine':
                tmp_1 = (1 - self.init_prob) * 0.5
                tmp_2 = math.pi * schedule_cur_steps
                tmp_3 = schedule_all_steps
                self.cur_prob = tmp_1 * (1 + math.cos(tmp_2 / tmp_3)) \
                    + self.init_prob
            else:
                raise ValueError('`prob_schedule` is eroor, it should be \
                    one of `linear` and `consine`.')

    def get_candidates_with_sample(self,
                                   num_samples: int) -> Tuple[Candidates, int]:
        """Get candidates with sampling from supernet and the candidates based
        on the current probablity."""
        num_sample_from_supernet = 0
        sampled_candidates = Candidates()
        for _ in range(num_samples):
            if random.random() >= self.cur_prob or len(self.candidates) == 0:
                subnet = self._sample_from_supernet()
                is_pass, _ = self._check_constraints(subnet)
                if is_pass:
                    sampled_candidates.append(subnet)
                num_sample_from_supernet += 1
            else:
                sampled_candidates.append(self._sample_from_candidates())
        return sampled_candidates, num_sample_from_supernet

    def update_candidates_scores(self) -> None:
        """Update candidates' scores, which are validated with the
        `dataloader_val`."""
        for i, candidate in enumerate(self.candidates.subnets):
            self.model.set_subnet(candidate)
            metrics = self._val_candidate()
            score = metrics[self.score_key] if len(metrics) != 0 else 0.
            self.candidates.set_resource(i, score, 'score')

    @torch.no_grad()
    def _val_candidate(self) -> Dict:
        """Run validation."""
        self.runner.model.eval()
        for data_batch in self.dataloader_val:
            outputs = self.runner.model.val_step(data_batch)
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        metrics = self.evaluator.evaluate(len(self.dataloader_val.dataset))
        return metrics

    def _sample_from_supernet(self) -> SupportRandomSubnet:
        """Sample from the supernet."""
        subnet = self.model.sample_subnet()
        return subnet

    def _sample_from_candidates(self) -> SupportRandomSubnet:
        """Sample from the candidates."""
        assert len(self.candidates) > 0
        subnet = random.choice(self.candidates.data)
        return subnet

    def _check_constraints(self, random_subnet: SupportRandomSubnet):
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

    def _save_candidates(self) -> None:
        """Save the candidates to init the next searching."""
        save_path = os.path.join(self.runner.work_dir, 'candidates.pkl')
        fileio.dump(self.candidates, save_path)
        self.runner.logger.info(f'candidates.pkl saved in '
                                f'{self.runner.work_dir}')
