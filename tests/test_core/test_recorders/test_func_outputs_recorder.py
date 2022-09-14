# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import FunctionOutputsRecorder


class TestFuncOutputsRecorder(TestCase):

    def test_init(self):

        _ = FunctionOutputsRecorder('toy_mod.toy_func')

        with self.assertRaisesRegex(TypeError, 'source should be'):
            _ = FunctionOutputsRecorder([1])

        with self.assertRaisesRegex(AssertionError, 'source must have at '):
            _ = FunctionOutputsRecorder('aaaaa')

        with self.assertRaisesRegex(ImportError, 'aaa is not imported'):
            _ = FunctionOutputsRecorder('aaa.bbb')

        with self.assertRaisesRegex(AssertionError, 'aaa is not in toy_mod'):
            _ = FunctionOutputsRecorder('toy_mod.aaa')

        with self.assertRaisesRegex(TypeError, 'TOY_VAR should be'):
            _ = FunctionOutputsRecorder('toy_mod.TOY_VAR')

    def test_context_manager(self):
        from toy_mod import execute_toy_func

        recorder = FunctionOutputsRecorder('toy_mod.toy_func')
        recorder.initialize()

        with recorder:
            execute_toy_func(1)

        data = recorder.get_record_data()
        self.assertTrue(data == 1)

        execute_toy_func(1)
        data = recorder.get_record_data()
        self.assertTrue(data == 1)
