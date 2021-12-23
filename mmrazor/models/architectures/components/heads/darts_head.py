# Copyright (c) OpenMMLab. All rights reserved.

from mmcls.models import HEADS, build_loss
from mmcls.models.heads import LinearClsHead
from torch import nn

from mmrazor.models.utils import add_prefix


@HEADS.register_module()
class DartsSubnetClsHead(LinearClsHead):

    def __init__(self, aux_in_channels, aux_loss, **kwargs):
        super(DartsSubnetClsHead, self).__init__(**kwargs)
        self.aux_linear = nn.Linear(aux_in_channels, self.num_classes)
        self.compute_aux_loss = build_loss(aux_loss)

    def forward_train(self, x, gt_label, **kwargs):

        losses = super().forward_train(x, gt_label, **kwargs)
        losses = add_prefix(losses, 'subnet')

        assert isinstance(x, tuple) and len(x) > 1
        x = x[0]

        cls_score = self.aux_linear(x)
        num_samples = len(cls_score)
        aux_loss = self.compute_aux_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)

        losses['aux_head.loss'] = aux_loss

        return losses
