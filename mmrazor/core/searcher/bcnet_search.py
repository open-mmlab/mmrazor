# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import SEARCHERS
from .evolution_search import EvolutionSearcher


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

        for loss, (subnet_l, subnet_r) in loss_rec:
            assert num_space_id == len(subnet_l)
            assert len(subnet_l) == len(subnet_r)
            for i, space_id in enumerate(sorted(subnet_l.keys())):
                out_mask = subnet_l[space_id]
                width = out_mask.sum().item()
                loss_matrix[i, width - self.min_channel_bins] += loss
                layer_width_cnt[i, width - self.min_channel_bins] += 1

        # for numerical stability, give width never chosen super large loss
        loss_matrix[layer_width_cnt == 0] = 1e-4
        loss_matrix /= (layer_width_cnt + 1e-5)

        # FLOPs calculation, layer flops is proportional to i/o channel width
        # FLOPs in different layers are different.
        F = torch.zeros((num_space_id, num_width, num_width), device=device)
        temp = torch.zeros((num_width, num_width), device=device)
        for row in range(num_width):
            for col in range(num_width):
                temp[row, col] = (row + 1) * (col + 1)
        temp /= self.min_channel_bins**2
        space_flops = self.algorithm.get_space_flops()
        for i, space_id in enumerate(sorted(space_flops.keys())):
            F[i, :, :] = space_flops[space_id] * temp
