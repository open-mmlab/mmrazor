# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmcls.models import ImageClassifier
except ImportError:
    from mmrazor.utils import get_placeholder
    ImageClassifier = get_placeholder('mmcls')


@TASK_UTILS.register_module()
class ImageClassifierPseudoLoss:
    """Calculate the pseudo loss to trace the topology of a `ImageClassifier`
    in MMClassification with `BackwardTracer`."""

    def __call__(self, model: ImageClassifier) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224)
        pseudo_output = model(pseudo_img)
        return sum(pseudo_output)
