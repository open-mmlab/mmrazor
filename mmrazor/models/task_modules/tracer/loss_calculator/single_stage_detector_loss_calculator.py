# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmdet.models import SingleStageDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    SingleStageDetector = get_placeholder('mmdet')


@TASK_UTILS.register_module()
class SingleStageDetectorPseudoLoss:
    """Calculate the pseudo loss to trace the topology of a
    `SingleStageDetector` in MMDetection with `BackwardTracer`.

    Args:
        input_shape (Tuple): The shape of the pseudo input. Defaults to
            (2, 3, 224, 224).
    """

    def __init__(self, input_shape=(2, 3, 224, 224)):
        self.input_shape = input_shape

    def __call__(self, model: SingleStageDetector) -> torch.Tensor:
        pseudo_img = torch.rand(self.input_shape)
        pseudo_output = model(pseudo_img)
        out = torch.tensor(0.)
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])

        return out
