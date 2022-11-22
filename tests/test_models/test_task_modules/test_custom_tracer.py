# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import CustomTracer, UntracedMethodRegistry
from mmrazor.testing import ConvBNReLU


class testCustomTracer(TestCase):

    def test_init(self):
        tracer = CustomTracer()
        assert tracer.skipped_methods.__len__() == 0

    def test_trace(self):
        tracer = CustomTracer()
        model = ConvBNReLU(3, 3, norm_cfg=dict(type='BN'))
        graph = tracer.trace(model)  # noqa: F841

    def test_auto_skip_call_module(self):
        pass

    def test_auto_skip_call_method(self):
        pass

    def test_configurable_skipped_methods(self):
        pass


class testUntracedMethodRgistry(TestCase):

    def test_init(self):
        self.assertEqual(len(UntracedMethodRegistry.method_dict), 0)

    def test_add_method(self):
        pass
