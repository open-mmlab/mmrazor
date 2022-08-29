# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict

from mmrazor.models import SelfDistill


class TestSelfDistill(TestCase):

    def test_init(self):

        student_recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))
        teacher_recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            distiller=dict(
                type='BYOTDistiller',
                student_recorders=student_recorders_cfg,
                teacher_recorders=teacher_recorders_cfg,
                distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_toy=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='conv')))))

        _ = SelfDistill(**alg_kwargs)

    def test_loss(self):

        student_recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))
        teacher_recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            distiller=dict(
                type='BYOTDistiller',
                student_recorders=student_recorders_cfg,
                teacher_recorders=teacher_recorders_cfg,
                distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_toy=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='conv')))))

        img = torch.randn(1, 3, 1, 1)

        alg = SelfDistill(**alg_kwargs)
        losses = alg(img, mode='loss')
        self.assertIn('distill.loss_toy', losses)
        self.assertIn('student.loss', losses)
