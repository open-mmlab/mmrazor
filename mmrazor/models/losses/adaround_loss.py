# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.logging import print_log

from mmrazor.registry import MODELS

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)


@MODELS.register_module()
class AdaRoundLoss(nn.Module):
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 weight: float = 1.,
                 iters: int = 10000,
                 beta_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.):
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(
            iters,
            warm_up=warm_up,
            start_beta=beta_range[0],
            end_beta=beta_range[1])
        self.count = 0

    def forward(self, subgraph, pred, tgt):
        """Compute the total loss for adaptive rounding: rec_loss is the
        quadratic output reconstruction loss, round_loss is a regularization
        term to optimize the rounding policy.

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """

        def lp_loss(pred, tgt, p=2.0):
            """loss function measured in L_p Norm."""
            return (pred - tgt).abs().pow(p).sum(1).mean()

        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        beta = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
        else:
            round_loss = 0
            for layer in subgraph.modules():
                if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += self.weight * (1 - (
                        (round_vals - .5).abs() * 2).pow(beta)).sum()

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print_log('Total loss:\t{:.3f} (rec_loss:{:.3f}, '
                      'round_loss:{:.3f})\tbeta={:.2f}\tcount={}'.format(
                          float(total_loss), float(rec_loss),
                          float(round_loss), beta, self.count))
        return total_loss


class LinearTempDecay:

    def __init__(self, t_max=10000, warm_up=0.2, start_beta=20, end_beta=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_beta = start_beta
        self.end_beta = end_beta

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_beta
        elif t > self.t_max:
            return self.end_beta
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_beta + (self.start_beta - self.end_beta) * \
                max(0.0, (1 - rel_t))
