# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

from mmengine import ConfigDict

from mmrazor.models import OverhaulFeatureDistillation
from .toy_models import ToyOFDStudent


class TestSingleTeacherDistill(TestCase):

    def test_init(self):

        recorders_cfg = ConfigDict(bn=dict(type='ModuleOutputs', source='bn'))

        alg_kwargs = ConfigDict(
            architecture=dict(type='ToyOFDStudent'),
            teacher=dict(type='ToyOFDTeacher'),
            distiller=dict(
                type='OFDDistiller',
                student_recorders=recorders_cfg,
                teacher_recorders=recorders_cfg,
                distill_losses=dict(loss_toy=dict(type='OFDLoss')),
                connectors=dict(loss_1_tfeat=dict(type='OFDTeacherConnector')),
                loss_forward_mappings=dict(
                    loss_toy=dict(
                        s_feature=dict(from_student=True, recorder='bn'),
                        t_feature=dict(
                            from_student=False,
                            recorder='bn',
                            connector='loss_1_tfeat')))))

        alg = OverhaulFeatureDistillation(**alg_kwargs)

        teacher = ToyOFDStudent()
        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teacher'] = teacher
        alg = OverhaulFeatureDistillation(**alg_kwargs_)
        self.assertEquals(alg.teacher, teacher)

        alg_kwargs_ = copy.deepcopy(alg_kwargs)
        alg_kwargs_['teacher'] = 'teacher'
        with self.assertRaisesRegex(TypeError,
                                    'teacher should be a `dict` or'):
            _ = OverhaulFeatureDistillation(**alg_kwargs_)
