# Copyright (c) OpenMMLab. All rights reserved.
import copy
from statistics import mode
import torch

from mmrazor.registry import TASK_UTILS
from mmseg.models.decode_heads.point_head import calculate_uncertainty, PointHead

try:
    from mmcls.models import ImageClassifier
except ImportError:
    from mmrazor.utils import get_placeholder
    ImageClassifier = get_placeholder('mmcls')

@TASK_UTILS.register_module()
class ImageSegPseudoLossGPU:
    """Calculate the pseudo loss to trace the topology of a `ImageClassifier`
    in MMClassification with `BackwardTracer`."""

    def __call__(self, model: ImageClassifier) -> torch.Tensor:
        pseudo_img = torch.rand(1, 3, 224, 224).cuda()
        point_coordinates = torch.Tensor([[[112, 112]]]).cuda()
        cfg = dict(
            num_points=1,
            oversample_ratio=1.0,
            importance_sample_ratio=1.0,
        )
        # NEED REVIEW: SYNCBATCHNORM only support on GPU
        model = model.cuda()
        model.train()
        #pseudo_output = model(pseudo_img)
        pseudo_output = model.backbone(pseudo_img)
        pseudo_output = model.neck(pseudo_output)
        # immutable decode_heads
        out = torch.tensor(0.).cuda()
        for levels in pseudo_output:
            out += sum([level.sum() for level in levels])
        model=model.cpu()
        return out
