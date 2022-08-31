# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.hub import get_model

from mmrazor.structures.tracer import ImageClassifierPseudoLoss


class TestLossCalculator(TestCase):

    def test_image_classifier_pseudo_loss(self):
        model = get_model(
            'mmcls::resnet/resnet34_8xb32_in1k.py', pretrained=False)
        loss_calculator = ImageClassifierPseudoLoss()
        loss = loss_calculator(model)
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0
