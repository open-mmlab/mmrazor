# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine import ConfigDict

from mmrazor.models import ConfigurableDistiller
from mmrazor.registry import MODELS


class ToyDistillLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, arg1, arg2):
        return arg1 + arg2


class TestConfigurableDistiller(TestCase):

    def setUp(self):
        MODELS.register_module(module=ToyDistillLoss, force=True)

    def tearDown(self):
        MODELS.module_dict.pop('ToyDistillLoss')

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

    def test_connector_list(self):
        recorders_cfg = ConfigDict(
            conv=dict(type='ModuleOutputs', source='conv'))
        norm_cfg = dict(type='BN', affine=False, track_running_stats=False)

        distiller_kwargs = ConfigDict(
            student_recorders=recorders_cfg,
            teacher_recorders=recorders_cfg,
            distill_losses=dict(loss_toy=dict(type='ToyDistillLoss')),
            loss_forward_mappings=dict(
                loss_toy=dict(
                    arg1=dict(
                        from_student=True,
                        recorder='conv',
                        connector='loss_1_sfeat'),
                    arg2=dict(from_student=False, recorder='conv'),
                )),
            connectors=dict(loss_1_sfeat=[
                dict(
                    type='ConvModuleConnector',
                    in_channel=3,
                    out_channel=4,
                    act_cfg=None),
                dict(type='NormConnector', norm_cfg=norm_cfg, in_channels=4)
            ]))

        distiller = ConfigurableDistiller(**distiller_kwargs)
        connectors = distiller.connectors
        self.assertIsInstance(connectors['loss_1_sfeat'], nn.Sequential)
