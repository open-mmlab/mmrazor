# Copyright (c) OpenMMLab. All rights reserved.
import copy
from statistics import mode
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmpose.model import TopdownPoseEstimator
except ImportError:
    from mmrazor.utils import get_placeholder
    ImageClassifier = get_placeholder('mmcls')

@TASK_UTILS.register_module()
class PosePseudoLoss:
    """Calculate the pseudo loss to trace the topology of a `ImageClassifier`
    in MMClassification with `BackwardTracer`."""

    def __call__(self, model: ImageClassifier) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224)
        pseudo_output = model.backbone(pseudo_img)
        # immutable decode_heads
        out = torch.tensor(0.).cuda()
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])
        model=model.cpu()
        return out
