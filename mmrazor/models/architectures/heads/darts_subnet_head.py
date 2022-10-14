# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import nn

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS

try:
    from mmcls.evaluation import Accuracy
    from mmcls.models.heads import LinearClsHead
    from mmcls.structures import ClsDataSample
except ImportError:
    from mmrazor.utils import get_placeholder
    Accuracy = get_placeholder('mmcls')
    LinearClsHead = get_placeholder('mmcls')
    ClsDataSample = get_placeholder('mmcls')


@MODELS.register_module()
class DartsSubnetClsHead(LinearClsHead):

    def __init__(self, aux_in_channels, aux_loss, **kwargs):
        super(DartsSubnetClsHead, self).__init__(**kwargs)
        self.aux_linear = nn.Linear(aux_in_channels, self.num_classes)
        self.aux_loss_module = MODELS.build(aux_loss)

    def forward_aux(self, feats: Tuple[torch.Tensor]):

        aux_feat = feats[0]
        aux_cls_score = self.aux_linear(aux_feat)
        return aux_cls_score

    def _get_aux_loss(self, cls_score: torch.Tensor,
                      data_samples: List[ClsDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            target = torch.hstack([i.gt_label.label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.aux_loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.
        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = super().loss(feats, data_samples, **kwargs)

        aux_cls_score = self.forward_aux(feats)
        aux_losses = self._get_aux_loss(aux_cls_score, data_samples, **kwargs)

        losses.update(add_prefix(aux_losses, 'aux_head.'))

        return losses
