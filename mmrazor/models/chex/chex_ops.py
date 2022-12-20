# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)


class ChexMixin:

    def prune_imp(self, num_remain):
        # compute channel importance for pruning
        return self._prune_imp(self.get_weight_matrix(), num_remain)

    @property
    def growth_imp(self):
        # compute channel importance for growth
        return self._growth_imp(self.get_weight_matrix())

    def get_weight_matrix(self):
        raise NotImplementedError()

    def _prune_imp(self, weight, num_remain):
        # weight: out * in. return the importance of each channel
        # modified from https://github.com/zejiangh/Filter-GaP
        assert num_remain <= weight.shape[0]
        weight_t = weight.T  # in out
        if weight_t.shape[0] >= weight_t.shape[1]:  # in >= out
            _, _, V = torch.svd(weight_t, some=True)  # out out
            Vk = V[:, :num_remain]  # out out'
            lvs = torch.norm(Vk, dim=1)  # out
            return lvs
        else:
            # l1-norm
            return weight.abs().mean(-1)

    def _growth_imp(self, weight):
        # weight: out * in. return the importance of each channel when growth

        def get_proj(weight):
            # out' in
            wt = weight.T  # in out'
            scatter = torch.matmul(wt.T, wt)  # out' out'
            inv = torch.pinverse(scatter)  # out' out'
            return torch.matmul(torch.matmul(wt, inv), wt.T)  # in in

        mask = self.get_mutable_attr('out_channels').current_mask
        n_mask = ~mask
        proj = get_proj(weight[mask])  # in in
        weight_c = weight[n_mask]  # out'' in

        error = (weight_c - weight_c @ proj).norm(dim=-1)
        all_errors = torch.zeros([weight.shape[0]], device=weight.device)
        all_errors.masked_scatter_(n_mask, error)
        return all_errors


class ChexConv2d(DynamicConv2d, ChexMixin):

    def get_weight_matrix(self):
        return self.weight.flatten(1)


class ChexLinear(DynamicLinear, ChexMixin):

    def get_weight_matrix(self):
        return self.weight
