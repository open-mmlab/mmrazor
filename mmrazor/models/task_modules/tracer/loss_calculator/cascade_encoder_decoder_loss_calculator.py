# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmseg.models import CascadeEncoderDecoder
except ImportError:
    from mmrazor.utils import get_placeholder
    CascadeEncoderDecoder = get_placeholder('mmseg')


@TASK_UTILS.register_module()
class CascadeEncoderDecoderPseudoLoss:
    """Calculate the pseudo loss to trace the topology of a
    `CascadeEncoderDecoder` in MMSegmentation with `BackwardTracer`."""

    def __call__(self, model: CascadeEncoderDecoder) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224)
        pseudo_output = model.backbone(pseudo_img)
        pseudo_output = model.neck(pseudo_output)
        # unmodified decode_heads
        out = torch.tensor(0.)
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])
        return out
