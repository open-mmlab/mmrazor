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
    in MMClassification with `BackwardTracer`.

    Args:
        input_shape (Tuple): The shape of the pseudo input. Defaults to
            (2, 3, 224, 224).
    """

    def __init__(self, input_shape=(2, 3, 224, 224)):
        self.input_shape = input_shape

    def __call__(self, model: ImageClassifier) -> torch.Tensor:
        pseudo_img = torch.rand(self.input_shape)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()
