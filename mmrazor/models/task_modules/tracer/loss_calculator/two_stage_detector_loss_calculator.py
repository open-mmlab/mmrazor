# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmdet.models import TwoStageDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    TwoStageDetector = get_placeholder('mmdet')


# todo: adapt to mmdet 2.0
@TASK_UTILS.register_module()
class TwoStageDetectorPseudoLoss:
    """Calculate the pseudo loss to trace the topology of a `TwoStageDetector`
    in MMDet with `BackwardTracer`."""

    def __call__(self, model: TwoStageDetector) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224)
        pseudo_output = model.backbone(pseudo_img)
        pseudo_output = model.neck(pseudo_output)
        out = torch.tensor(0.)
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])

        return out
