# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmpose.models import TopdownPoseEstimator
except ImportError:
    from mmrazor.utils import get_placeholder
    TopdownPoseEstimator = get_placeholder('mmpose')


@TASK_UTILS.register_module()
class TopdownPoseEstimatorPseudoLoss:
    """Calculate the pseudo loss to trace the topology of a
    `TopdownPoseEstimator` in MMPose with `BackwardTracer`."""

    def __call__(self, model: TopdownPoseEstimator) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224)
        pseudo_output = model.backbone(pseudo_img)
        # immutable decode_heads
        out = torch.tensor(0.)
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])
        return out
