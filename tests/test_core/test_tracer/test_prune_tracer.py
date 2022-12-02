# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor import digit_version
from mmrazor.models.task_modules.tracer import PruneTracer
from ...data.models import SingleLineModel


class TestPruneTracer(TestCase):

    def test_backward_tracer(self):
        model = SingleLineModel()
        tracer = PruneTracer(tracer_type='BackwardTracer')
        unit_configs = tracer.trace(model)
        print(unit_configs)

    def test_fx_tracer(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('torch<1.12.0')
        model = SingleLineModel()
        tracer = PruneTracer(tracer_type='FxTracer')
        unit_configs = tracer.trace(model)
        print(unit_configs)
