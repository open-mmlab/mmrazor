# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp

import mmcv
import torch
from mmcv.runner import get_dist_info

from ..builder import SEARCHERS
from ..utils import broadcast_object_list


@SEARCHERS.register_module()
class GreedySearcher():
    """Search with the greedy algorithm.

    We start with the largest model and compare the network accuracy among
    the architectures where each layer is slimmed by one channel bin. We then
    greedily slim the layer with minimal performance drop. During the iterative
    slimming, we obtain optimized channel configurations under different
    resource constraints. We stop until reaching the strictest constraint
    (e.g., 200M FLOPs).

    Args:
        algorithm (:obj:`torch.nn.Module`): Specific implemented algorithm
         based specific task in mmRazor, eg: AutoSlim.
        dataloader (:obj:`torch.nn.Dataloader`): Pytorch data loader.
        target_flops (list): The target flops of the searched models.
        test_fn (callable): test a model with samples from a dataloader,
            and return the test results.
        work_dir (str): Output result file.
        logger (logging.Logger): To log info in search stage.
        max_channel_bins (int): The maximum number of channel bins in each
            layer. Note that each layer is slimmed by one channel bin.
        min_channel_bins (int): The minimum number of channel bins in each
            layer. Default to 1.
        metrics (str | list[str]): Metrics to be evaluated.
            Default value is ``accuracy``
        metric_options (dict, optional): Options for calculating metrics.
            Allowed keys are 'topk', 'thrs' and 'average_mode'.
            Defaults to None.
        score_key (str): The metric to judge the performance of a model.
            Defaults to `accuracy_top-1`.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching. Defaults to None.
    """

    def __init__(self,
                 algorithm,
                 dataloader,
                 target_flops,
                 test_fn,
                 work_dir,
                 logger,
                 max_channel_bins,
                 min_channel_bins=1,
                 metrics='accuracy',
                 metric_options=None,
                 score_key='accuracy_top-1',
                 resume_from=None,
                 **search_kwargs):
        super(GreedySearcher, self).__init__()
        if not hasattr(algorithm, 'module'):
            raise NotImplementedError('Do not support searching with cpu.')

        self.algorithm = algorithm.module
        self.algorithm_for_test = algorithm
        self.dataloader = dataloader
        self.target_flops = sorted(target_flops, reverse=True)
        self.test_fn = test_fn
        self.work_dir = work_dir
        self.logger = logger
        self.max_channel_bins = max_channel_bins
        self.min_channel_bins = min_channel_bins
        self.metrics = metrics
        self.metric_options = metric_options
        self.score_key = score_key
        self.resume_from = resume_from

    def search(self):
        """Greedy Slimming."""
        algorithm = self.algorithm

        rank, _ = get_dist_info()

        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)  # a dict
            result_subnet = searcher_resume['result_subnet']
            result_flops = searcher_resume['result_flops']
            subnet = searcher_resume['subnet']
            flops = searcher_resume['flops']
            self.logger.info(f'Resume from subnet: {subnet}')
        else:
            result_subnet, result_flops = [], []
            # We start with the largest model
            algorithm.pruner.set_max_channel()
            max_subnet = algorithm.pruner.get_max_channel_bins(
                self.max_channel_bins)
            # channel_cfg
            subnet = max_subnet
            flops = algorithm.get_subnet_flops()

        for target in self.target_flops:
            if self.resume_from is not None and flops <= target:
                continue

            if flops <= target:
                algorithm.pruner.set_channel_bins(subnet,
                                                  self.max_channel_bins)
                channel_cfg = algorithm.pruner.export_subnet()
                result_subnet.append(channel_cfg)
                result_flops.append(flops)
                self.logger.info(f'Find model flops {flops} <= {target}')
                continue

            while flops > target:
                # search which layer needs to shrink
                best_score = None
                best_subnet = None

                # During distributed training, the order of ``subnet.keys()``
                # on different ranks may be different. So we need to sort it
                # first.
                for i, name in enumerate(sorted(subnet.keys())):
                    new_subnet = copy.deepcopy(subnet)
                    # we prune the very last channel bin
                    last_bin_ind = torch.where(new_subnet[name] == 1)[0][-1]
                    # The ``new_subnet`` on different ranks are the same,
                    # so we do not need to broadcast here.
                    new_subnet[name][last_bin_ind] = 0
                    if torch.sum(new_subnet[name]) < self.min_channel_bins:
                        # subnet is invalid
                        continue

                    algorithm.pruner.set_channel_bins(new_subnet,
                                                      self.max_channel_bins)

                    outputs = self.test_fn(self.algorithm_for_test,
                                           self.dataloader)
                    broadcast_scores = [None]
                    if rank == 0:
                        eval_result = self.dataloader.dataset.evaluate(
                            outputs, self.metrics, self.metric_options)
                        broadcast_scores = [eval_result[self.score_key]]

                    # Broadcasts scores in broadcast_scores to the whole
                    # group.
                    broadcast_scores = broadcast_object_list(broadcast_scores)
                    score = broadcast_scores[0]
                    self.logger.info(
                        f'Slimming group {name}, {self.score_key}: {score}')
                    if best_score is None or score > best_score:
                        best_score = score
                        best_subnet = new_subnet

                if best_subnet is None:
                    raise RuntimeError(
                        'Cannot find any valid model, check your '
                        'configurations.')

                subnet = best_subnet
                algorithm.pruner.set_channel_bins(subnet,
                                                  self.max_channel_bins)
                flops = algorithm.get_subnet_flops()
                self.logger.info(
                    f'Greedy find model, score: {best_score}, FLOPS: {flops}')

                save_for_resume = dict()
                save_for_resume['result_subnet'] = result_subnet
                save_for_resume['result_flops'] = result_flops
                save_for_resume['subnet'] = subnet
                save_for_resume['flops'] = flops
                mmcv.fileio.dump(save_for_resume,
                                 osp.join(self.work_dir, 'latest.pkl'))

            algorithm.pruner.set_channel_bins(subnet, self.max_channel_bins)
            channel_cfg = algorithm.pruner.export_subnet()
            result_subnet.append(channel_cfg)
            result_flops.append(flops)
            self.logger.info(f'Find model flops {flops} <= {target}')

        self.logger.info('Search models done.')

        if rank == 0:
            for flops, subnet in zip(result_flops, result_subnet):
                mmcv.fileio.dump(
                    subnet,
                    os.path.join(self.work_dir,
                                 'subnet_{}.yaml'.format(flops)))
            self.logger.info(f'Save searched results to {self.work_dir}')
