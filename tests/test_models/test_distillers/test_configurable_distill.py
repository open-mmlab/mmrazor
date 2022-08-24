# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

from mmengine import ConfigDict

from mmrazor.models import ConfigurableDistiller


class TestConfigurableDistiller(TestCase):

    def test_init(self):

        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))

        distiller_kwargs = ConfigDict(
            student_recorders=recorders_cfg,
            teacher_recorders=recorders_cfg,
            distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
            loss_forward_mappings=dict(
                loss_toy=dict(
                    arg1=dict(from_student=True, recorder='conv'),
                    arg2=dict(from_student=False, recorder='conv'),
                )),
        )

        _ = ConfigurableDistiller(**distiller_kwargs)

        distiller_kwargs_ = copy.deepcopy(distiller_kwargs)
        distiller_kwargs_['distill_losses'] = None
        with self.assertRaisesRegex(AssertionError,
                                    '"loss_toy" is not in distill'):
            _ = ConfigurableDistiller(**distiller_kwargs_)

        distiller_kwargs_ = copy.deepcopy(distiller_kwargs)
        distiller_kwargs_['distill_losses'] = dict(
            toy=dict(type='ToyDistillLoss'))
        distiller_kwargs_['loss_forward_mappings'] = dict(
            toy=dict(
                arg1=dict(from_student=True, recorder='conv'),
                arg2=dict(from_student=False, recorder='conv')))
        with self.assertWarnsRegex(UserWarning, 'Warning: If toy is a'):
            _ = ConfigurableDistiller(**distiller_kwargs_)

        distiller_kwargs_ = copy.deepcopy(distiller_kwargs)
        distiller_kwargs_['loss_forward_mappings'] = None
        _ = ConfigurableDistiller(**distiller_kwargs_)

        distiller_kwargs_ = copy.deepcopy(distiller_kwargs)
        distiller_kwargs_['loss_forward_mappings'] = list('AAA')

        with self.assertRaisesRegex(TypeError,
                                    'loss_forward_mappings should be '):
            _ = ConfigurableDistiller(**distiller_kwargs_)

        distiller_kwargs_ = copy.deepcopy(distiller_kwargs)
        distiller_kwargs_['loss_forward_mappings']['loss_toy'] = list()
        with self.assertRaisesRegex(
                TypeError, 'Each item of loss_forward_mappings should be '):
            _ = ConfigurableDistiller(**distiller_kwargs_)

        distiller_kwargs_ = copy.deepcopy(distiller_kwargs)
        distiller_kwargs_.loss_forward_mappings.loss_toy.arg1.from_student = ''
        with self.assertRaisesRegex(TypeError,
                                    'from_student should be a bool'):
            _ = ConfigurableDistiller(**distiller_kwargs_)
