# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

from mmcv import ConfigDict
from toy_models import ToyStudent

from mmrazor.models import ConfigurableDistill


class TestConfigurableDistill(TestCase):

    def test_init(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        student = ToyStudent()

        alg_kwargs = ConfigDict(
            architecture=student,
            student_recorders=recorders_cfg,
            teacher_recorders=recorders_cfg,
            distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
            loss_forward_mappings=dict(
                loss_toy=dict(
                    arg1=dict(from_student=True, recorder='conv'),
                    arg2=dict(from_student=False, recorder='conv'),
                )),
        )

        alg = ConfigurableDistill(**alg_kwargs)
        self.assertEquals(alg.student, alg.architecture)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['distill_losses'] = None
        with self.assertRaisesRegex(AssertionError,
                                    '"loss_toy" is not in distill'):
            _ = ConfigurableDistill(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['distill_losses'] = dict(toy=dict(type='ToyDistillLoss'))
        alg_kwargs_['loss_forward_mappings'] = dict(
            toy=dict(
                arg1=dict(from_student=True, recorder='conv'),
                arg2=dict(from_student=False, recorder='conv')))
        with self.assertWarnsRegex(UserWarning, 'Warning: If toy is a'):
            _ = ConfigurableDistill(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['loss_forward_mappings'] = None
        _ = ConfigurableDistill(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['loss_forward_mappings'] = list('AAA')

        with self.assertRaisesRegex(TypeError,
                                    'loss_forward_mappings should be '):
            _ = ConfigurableDistill(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['loss_forward_mappings']['loss_toy'] = list()
        with self.assertRaisesRegex(
                TypeError, 'Each item of loss_forward_mappings should be '):
            _ = ConfigurableDistill(**alg_kwargs_)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_.loss_forward_mappings.loss_toy.arg1.from_student = ''
        with self.assertRaisesRegex(TypeError,
                                    'from_student should be a bool'):
            _ = ConfigurableDistill(**alg_kwargs_)
