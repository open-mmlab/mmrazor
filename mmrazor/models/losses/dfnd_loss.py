# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class DFNDLoss(nn.Module):
    """Loss function for DFND.
       https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Student_Networks_in_the_Wild_CVPR_2021_paper.pdf

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
        num_classes (int): Number of classes.
        teacher_acc (float): The performance of teacher network in the target
             dataset.
        batch_select (float): ratio of data in the wild dataset to participate
             in training.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
        num_classes: int = 1000,
        teacher_acc: float = 0.95,
        batch_select: float = 0.5,
    ):
        super(DFNDLoss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.noisy_adaptation = torch.nn.Parameter(
            torch.zeros(num_classes, num_classes - 1))
        self.teacher_acc = teacher_acc
        self.num_classes = num_classes
        self.nll_loss = torch.nn.NLLLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.batch_select = batch_select

    def noisy(self):
        noise_adaptation_softmax = torch.nn.functional.softmax(
            self.noisy_adaptation, dim=1) * (1 - self.teacher_acc)
        noise_adaptation_layer = torch.zeros(self.num_classes,
                                             self.num_classes).to(
                                                 self.noisy_adaptation.device)
        tc = torch.FloatTensor([self.teacher_acc
                                ]).to(noise_adaptation_softmax.device)
        for i in range(self.num_classes):
            if i == 0:
                noise_adaptation_layer[i] = \
                    torch.cat([tc, noise_adaptation_softmax[i][i:]])
            if i == self.num_classes - 1:
                noise_adaptation_layer[i] = \
                    torch.cat([noise_adaptation_softmax[i][:i], tc])
            else:
                noise_adaptation_layer[i] = \
                    torch.cat([noise_adaptation_softmax[i][:i], tc,
                               noise_adaptation_softmax[i][i:]])
        return noise_adaptation_layer

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        pred = preds_T.data.max(1)[1]
        loss_t = self.ce_loss(preds_T, pred)
        positive_loss_idx = loss_t.topk(
            int(self.batch_select * preds_S.shape[0]), largest=False)[1]
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        log_softmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        softmax_preds_S_adaptation = torch.matmul(
            F.softmax(preds_S, dim=1), self.noisy())
        loss = (self.tau**2) * (
            torch.sum(
                F.kl_div(
                    log_softmax_preds_S[positive_loss_idx],
                    softmax_pred_T[positive_loss_idx],
                    reduction='none')) / preds_S.shape[0])
        loss += self.nll_loss(torch.log(softmax_preds_S_adaptation), pred)
        return self.loss_weight * loss
