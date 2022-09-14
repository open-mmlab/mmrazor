# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.hub import get_model

from mmrazor.models.task_modules.tracer import (ImageClassifierPseudoLoss,
                                                SingleStageDetectorPseudoLoss)


class TestLossCalculator(TestCase):

    def test_image_classifier_pseudo_loss(self):
        model = get_model(
            'mmcls::resnet/resnet34_8xb32_in1k.py', pretrained=False)
        loss_calculator = ImageClassifierPseudoLoss()
        loss = loss_calculator(model)
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

    def test_single_stage_detector_pseudo_loss(self):
        model = get_model(
            'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py', pretrained=False)
        loss_calculator = SingleStageDetectorPseudoLoss()
        loss = loss_calculator(model)
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0
