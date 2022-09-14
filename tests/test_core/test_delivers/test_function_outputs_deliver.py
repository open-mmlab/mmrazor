# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import FunctionOutputsDelivery


class TestFuncOutputsDeliver(TestCase):

    def test_init(self):

        with self.assertRaisesRegex(TypeError, 'func_path should be'):
            _ = FunctionOutputsDelivery(max_keep_data=1, func_path=1)

        with self.assertRaisesRegex(AssertionError, 'func_path must have at '):
            _ = FunctionOutputsDelivery(max_keep_data=1, func_path='toy_func')

        with self.assertRaisesRegex(ImportError, 'aaa is not imported'):
            _ = FunctionOutputsDelivery(max_keep_data=1, func_path='aaa.bb')

        with self.assertRaisesRegex(AssertionError, 'bb is not in toy_mod'):
            _ = FunctionOutputsDelivery(
                max_keep_data=1, func_path='toy_module.bb')

        with self.assertRaisesRegex(TypeError, 'TOY_VAR should be'):
            _ = FunctionOutputsDelivery(
                max_keep_data=1, func_path='toy_module.TOY_VAR')

    def test_context_manager(self):
        import toy_module

        delivery = FunctionOutputsDelivery(
            max_keep_data=2, func_path='toy_module.toy_func')

        delivery.override_data = False
        with delivery:
            output1_tea = toy_module.toy_func()
            output2_tea = toy_module.toy_func()

        with self.assertRaisesRegex(AssertionError, 'push into an full queue'):
            with delivery:
                _ = toy_module.toy_func()

        delivery.override_data = True
        with delivery:
            output1_stu = toy_module.toy_func()
            output2_stu = toy_module.toy_func()

        # With ``FunctionOutputsDeliver``, outputs of the teacher and
        # the student are the same.
        assert output1_stu == output1_tea and output2_stu == output2_tea

        with self.assertRaisesRegex(AssertionError, 'pop from an empty queue'):
            with delivery:
                _ = toy_module.toy_func()
