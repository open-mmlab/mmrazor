# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class CRDLoss(nn.Module):
    """Variate CRD Loss, ICLR 2020.

    https://arxiv.org/abs/1910.10699
    Args:
        loss_weight (float, optional): loss weight. Defaults to 1.0.
        temperature (float, optional): temperature. Defaults to 0.07.
        neg_num (int, optional): number of negative samples. Defaults to 16384.
        sample_n (int, optional): number of total samples. Defaults to 50000.
        dim_out (int, optional): output channels. Defaults to 128.
        momentum (float, optional): momentum. Defaults to 0.5.
        eps (double, optional): eps. Defaults to 1e-7.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 temperature=0.07,
                 neg_num=16384,
                 sample_n=50000,
                 dim_out=128,
                 momentum=0.5,
                 eps=1e-7):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

        self.contrast = ContrastMemory(dim_out, sample_n, neg_num, temperature,
                                       momentum)
        self.criterion_s_t = ContrastLoss(sample_n, eps=self.eps)

    def forward(self, s_feats, t_feats, data_samples):
        input_data = data_samples[0]
        assert 'sample_idx' in input_data, \
            'you should pass a dict with key `sample_idx` in mimic function.'
        assert isinstance(
            input_data.sample_idx, torch.Tensor
        ), f'`sample_idx` must be a tensor, but get {type(input_data.sample_idx)}'  # noqa: E501

        sample_idxs = torch.stack(
            [sample.sample_idx for sample in data_samples])
        if 'contrast_sample_idxs' in input_data:
            assert isinstance(
                input_data.contrast_sample_idxs, torch.Tensor
            ), f'`contrast_sample_idxs` must be a tensor, but get {type(input_data.contrast_sample_idxs)}'  # noqa: E501
            contrast_sample_idxs = torch.stack(
                [sample.contrast_sample_idxs for sample in data_samples])
        else:
            contrast_sample_idxs = None
        out_s, out_t = self.contrast(s_feats, t_feats, sample_idxs,
                                     contrast_sample_idxs)
        s_loss = self.criterion_s_t(out_s)
        t_loss = self.criterion_s_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """contrastive loss, corresponding to Eq (18)

    Args:
        n_data (int): number of data
        eps (float, optional): eps. Defaults to 1e-7.
    """

    def __init__(self, n_data: int, eps: float = 1e-7):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data
        self.eps = eps

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + self.eps)).log_()

        # loss for neg_sample negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
                           P_neg.add(m * Pn + self.eps)).log_()

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class ContrastMemory(nn.Module):
    """memory buffer that supplies large amount of negative samples.

    https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py

    Args:
        dim_out (int, optional): output channels. Defaults to 128.
        n_sample (int, optional): number of total samples.
            Defaults to 50000.
        neg_sample (int, optional): number of negative samples.
            Defaults to 16384.
        T (float, optional): temperature. Defaults to 0.07.
        momentum (float, optional): momentum. Defaults to 0.5.
    """

    def __init__(self,
                 dim_out: int,
                 n_sample: int,
                 neg_sample: int,
                 T: float = 0.07,
                 momentum: float = 0.5):
        super(ContrastMemory, self).__init__()
        self.n_sample = n_sample
        self.unigrams = torch.ones(self.n_sample)
        self.multinomial = AliasMethod(self.unigrams)
        # self.multinomial.cuda()
        self.neg_sample = neg_sample

        self.register_buffer('params',
                             torch.tensor([neg_sample, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(dim_out / 3)
        self.register_buffer(
            'memory_v1',
            torch.rand(n_sample, dim_out).mul_(2 * stdv).add_(-stdv))
        self.register_buffer(
            'memory_v2',
            torch.rand(n_sample, dim_out).mul_(2 * stdv).add_(-stdv))

    def forward(self,
                feat_s: torch.Tensor,
                feat_t: torch.Tensor,
                idx: torch.Tensor,
                sample_idx: Union[None, torch.Tensor] = None) -> torch.Tensor:
        neg_sample = int(self.params[0].item())
        T = self.params[1].item()
        Z_s = self.params[2].item()
        Z_t = self.params[3].item()

        momentum = self.params[4].item()
        bsz = feat_s.size(0)
        n_sample = self.memory_v1.size(0)
        dim_out = self.memory_v1.size(1)

        # original score computation
        if sample_idx is None:
            sample_idx = self.multinomial.draw(bsz * (self.neg_sample + 1))\
                .view(bsz, -1)
            sample_idx.select(1, 0).copy_(idx.data)
        # sample
        weight_s = torch.index_select(self.memory_v1, 0,
                                      sample_idx.view(-1)).detach()
        weight_s = weight_s.view(bsz, neg_sample + 1, dim_out)
        out_t = torch.bmm(weight_s, feat_t.view(bsz, dim_out, 1))
        out_t = torch.exp(torch.div(out_t, T))
        # sample
        weight_t = torch.index_select(self.memory_v2, 0,
                                      sample_idx.view(-1)).detach()
        weight_t = weight_t.view(bsz, neg_sample + 1, dim_out)
        out_s = torch.bmm(weight_t, feat_s.view(bsz, dim_out, 1))
        out_s = torch.exp(torch.div(out_s, T))

        # set Z if haven't been set yet
        if Z_s < 0:
            self.params[2] = out_s.mean() * n_sample
            Z_s = self.params[2].clone().detach().item()
            print('normalization constant Z_s is set to {:.1f}'.format(Z_s))
        if Z_t < 0:
            self.params[3] = out_t.mean() * n_sample
            Z_t = self.params[3].clone().detach().item()
            print('normalization constant Z_t is set to {:.1f}'.format(Z_t))

        # compute out_s, out_t
        out_s = torch.div(out_s, Z_s).contiguous()
        out_t = torch.div(out_t, Z_t).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, idx.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(feat_s, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, idx, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, idx.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(feat_t, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, idx, updated_v2)

        return out_s, out_t


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/
    the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    Args:
        probs (torch.Tensor): probility vector.
    """

    def __init__(self, probs: torch.Tensor) -> None:

        if probs.sum() > 1:
            probs.div_(probs.sum())
        neg_sample = len(probs)
        self.prob = torch.zeros(neg_sample)
        self.alias = torch.LongTensor([0] * neg_sample)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/neg_sample.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = neg_sample * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        """To cuda device."""
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N: int) -> torch.Tensor:
        """Draw N samples from multinomial."""
        neg_sample = self.alias.size(0)

        kk = torch.zeros(
            N, dtype=torch.long,
            device=self.prob.device).random_(0, neg_sample)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj
