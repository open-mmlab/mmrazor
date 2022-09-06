# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import MethodOutputsRecorder


class TestFuncOutputsRecorder(TestCase):

    def test_init(self):

        _ = MethodOutputsRecorder('toy_mod.ToyClass.toy')

        with self.assertRaisesRegex(TypeError, 'source should be'):
            _ = MethodOutputsRecorder([1])

        with self.assertRaisesRegex(AssertionError, 'source must have at '):
            _ = MethodOutputsRecorder('aaaaa')

        with self.assertRaisesRegex(AssertionError, 'source must have at '):
            _ = MethodOutputsRecorder('aaa.bbb')

        with self.assertRaisesRegex(ImportError, 'aaa is not imported'):
            _ = MethodOutputsRecorder('aaa.bbb.ccc')

        with self.assertRaisesRegex(AssertionError, 'aaa is not in toy_mod'):
            _ = MethodOutputsRecorder('toy_mod.aaa.bbb')

        with self.assertRaisesRegex(TypeError, 'toy_func should be'):
            _ = MethodOutputsRecorder('toy_mod.toy_func.bbb')

        with self.assertRaisesRegex(AssertionError, 'bbb is not in ToyClass'):
            _ = MethodOutputsRecorder('toy_mod.ToyClass.bbb')

        with self.assertRaisesRegex(TypeError, 'TOY_CLS should be'):
            _ = MethodOutputsRecorder('toy_mod.ToyClass.TOY_CLS')

    def test_context_manager(self):
        from toy_mod import ToyClass

        toy = ToyClass()

        recorder = MethodOutputsRecorder('toy_mod.ToyClass.toy')
        recorder.initialize()

        with recorder:
            result = toy.toy()

        data = recorder.get_record_data()
        self.assertTrue(data == result)

        result_ = toy.toy()

        data = recorder.get_record_data()
        self.assertTrue(data == result)
        self.assertFalse(result_ == result)
