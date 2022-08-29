# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from mmengine import ConfigDict
from mmengine.optim import build_optim_wrapper

from mmrazor.models import DAFLDataFreeDistillation, DataFreeDistillation


class TestDataFreeDistill(TestCase):

    def test_init(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            teachers=dict(
                tea1=dict(build_cfg=dict(type='ToyTeacher')),
                tea2=dict(build_cfg=dict(type='ToyTeacher'))),
            generator=dict(type='ToyGenerator'),
            distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea1_conv=dict(type='ModuleOutputs', source='tea1.conv')),
                distill_losses=dict(loss_dis=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_dis=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea1_conv')))),
            generator_distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea2_conv=dict(type='ModuleOutputs', source='tea2.conv')),
                distill_losses=dict(loss_gen=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_gen=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea2_conv')))),
        )

        alg = DataFreeDistillation(**alg_kwargs)
        self.assertEquals(len(alg.teachers), len(alg_kwargs['teachers']))

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teachers'] = 'ToyTeacher'
        with self.assertRaisesRegex(TypeError,
                                    'teacher should be a `dict` but got '):
            alg = DataFreeDistillation(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['generator'] = 'ToyGenerator'
        with self.assertRaisesRegex(
                TypeError, 'generator should be a `dict` instance, but got '):
            _ = DataFreeDistillation(**alg_kwargs_)

    def test_loss(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            teachers=dict(
                tea1=dict(build_cfg=dict(type='ToyTeacher')),
                tea2=dict(build_cfg=dict(type='ToyTeacher'))),
            generator=dict(type='ToyGenerator'),
            distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea1_conv=dict(type='ModuleOutputs', source='tea1.conv')),
                distill_losses=dict(loss_dis=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_dis=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea1_conv')))),
            generator_distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea2_conv=dict(type='ModuleOutputs', source='tea2.conv')),
                distill_losses=dict(loss_gen=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_gen=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea2_conv')))),
        )

        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD', lr=0.1, weight_decay=0.01, momentum=0.9))

        data = dict(inputs=torch.randn(3, 1, 1), data_samples=None)

        alg = DataFreeDistillation(**alg_kwargs)
        optim_wrapper = build_optim_wrapper(alg, optim_wrapper_cfg)
        optim_wrapper_dict = dict(
            architecture=optim_wrapper, generator=optim_wrapper)

        losses = alg.train_step(data, optim_wrapper_dict)
        self.assertIn('distill.loss_dis', losses)
        self.assertIn('distill.loss', losses)
        self.assertIn('generator.loss_gen', losses)
        self.assertIn('generator.loss', losses)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['student_iter'] = 5
        alg = DataFreeDistillation(**alg_kwargs_)
        losses = alg.train_step(data, optim_wrapper_dict)
        self.assertIn('distill.loss_dis', losses)
        self.assertIn('distill.loss', losses)
        self.assertIn('generator.loss_gen', losses)
        self.assertIn('generator.loss', losses)


class TestDAFLDataFreeDistill(TestCase):

    def test_init(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            teachers=dict(
                tea1=dict(build_cfg=dict(type='ToyTeacher')),
                tea2=dict(build_cfg=dict(type='ToyTeacher'))),
            generator=dict(type='ToyGenerator'),
            distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea1_conv=dict(type='ModuleOutputs', source='tea1.conv')),
                distill_losses=dict(loss_dis=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_dis=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea1_conv')))),
            generator_distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea2_conv=dict(type='ModuleOutputs', source='tea2.conv')),
                distill_losses=dict(loss_gen=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_gen=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea2_conv')))))

        alg = DAFLDataFreeDistillation(**alg_kwargs)
        self.assertEquals(len(alg.teachers), len(alg_kwargs['teachers']))

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teachers'] = 'ToyTeacher'
        with self.assertRaisesRegex(TypeError,
                                    'teacher should be a `dict` but got '):
            alg = DAFLDataFreeDistillation(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['generator'] = 'ToyGenerator'
        with self.assertRaisesRegex(
                TypeError, 'generator should be a `dict` instance, but got '):
            _ = DAFLDataFreeDistillation(**alg_kwargs_)

    def test_loss(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyStudent'),
            teachers=dict(
                tea1=dict(build_cfg=dict(type='ToyTeacher')),
                tea2=dict(build_cfg=dict(type='ToyTeacher'))),
            generator=dict(type='ToyGenerator'),
            distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea1_conv=dict(type='ModuleOutputs', source='tea1.conv')),
                distill_losses=dict(loss_dis=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_dis=dict(
                        arg1=dict(from_student=True, recorder='conv'),
                        arg2=dict(from_student=False, recorder='tea1_conv')))),
            generator_distiller=dict(
                type='ConfigurableDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=dict(
                    tea1_conv=dict(type='ModuleOutputs', source='tea1.conv'),
                    tea2_conv=dict(type='ModuleOutputs', source='tea2.conv')),
                distill_losses=dict(loss_gen=dict(type='ToyDistillLoss')),
                loss_forward_mappings=dict(
                    loss_gen=dict(
                        arg1=dict(from_student=False, recorder='tea1_conv'),
                        arg2=dict(from_student=False, recorder='tea2_conv')))))

        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD', lr=0.1, weight_decay=0.01, momentum=0.9))

        data = dict(inputs=torch.randn(3, 1, 1), data_samples=None)

        alg = DAFLDataFreeDistillation(**alg_kwargs)
        optim_wrapper = build_optim_wrapper(alg, optim_wrapper_cfg)
        optim_wrapper_dict = dict(
            architecture=optim_wrapper, generator=optim_wrapper)

        losses = alg.train_step(data, optim_wrapper_dict)
        self.assertIn('distill.loss_dis', losses)
        self.assertIn('distill.loss', losses)
        self.assertIn('generator.loss_gen', losses)
        self.assertIn('generator.loss', losses)
