# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS

try:
    from mmdet.models.detectors import BaseDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDetector = get_placeholder('mmdet')


# todo: adapt to mmdet 2.0
@TASK_UTILS.register_module()
class SingleStageDetectorPseudoLoss:

    def __call__(self, model: BaseDetector) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224)
        pseudo_output = model.forward_dummy(pseudo_img)
        out = torch.tensor(0.)
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])

        return out
