# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from mmengine import ConfigDict

from mmrazor.models import SingleTeacherDistill
from .toy_models import ToyStudent


class TestSingleTeacherDistill(TestCase):

    def test_init(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            teacher=dict(type='ToyTeacher'),
            distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=recorders_cfg,
                distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_toy=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='conv')))))

        alg = SingleTeacherDistill(**alg_kwargs)

        teacher = ToyStudent()
        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teacher'] = teacher
        alg = SingleTeacherDistill(**alg_kwargs_)
        self.assertEquals(alg.teacher, teacher)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teacher'] = 'teacher'
        with self.assertRaisesRegex(TypeError,
                                    'teacher should be a `dict` or'):
            _ = SingleTeacherDistill(**alg_kwargs_)

    def test_loss(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            teacher=dict(type='ToyTeacher'),
            distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=recorders_cfg,
                distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_toy=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='conv')))))

        img = torch.randn(1, 3, 1, 1)

        alg = SingleTeacherDistill(**alg_kwargs)
        losses = alg(img, mode='loss')
        self.assertIn('distill.loss_toy', losses)
        self.assertIn('student.loss', losses)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teacher_trainable'] = True
        alg = SingleTeacherDistill(**alg_kwargs_)
        losses = alg(img, mode='loss')
        self.assertIn('distill.loss_toy', losses)
        self.assertIn('student.loss', losses)
        self.assertIn('teacher.loss', losses)
