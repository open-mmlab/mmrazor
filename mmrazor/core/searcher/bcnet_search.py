# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
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
                # channels to channel_bins
                width = round(out_mask.sum() * self.max_channel_bins / out_mask.numel())
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

        # output sample possibility is softmax of P
        P = torch.autograd.Variable(torch.randn_like(loss_matrix), requires_grad=True)
        optim = torch.optim.SGD([P], lr=0.01)

        for _ in range(100000):
            optim.zero_grad()
            prob = F.softmax(P, dim=1)
            prob_shift = F.pad(prob, (0, 0, 0, 1), value=1.0 / num_width)[1:, :]
            z = (prob * loss_matrix).sum(dim=1)

            F_e = (F * prob.view(num_space_id, num_width, 1) * prob_shift.view(num_space_id, 1, num_width)).sum()
            loss = z.mean() + (1.0 - F_e / self.flops_limit) ** 2
            loss.backward()
            optim.step()
            if _ % 10000 == 0:
                self.logger.info(f'Initialize Prior Population: Epoch {_} Loss {loss.item()}')

        P.detach_()
        prob = F.softmax(P, dim=1).cpu().numpy()
        self.logger.info(f'Initialize Prior Population Done: P {prob}')

        for _ in range(self.population_num):
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
