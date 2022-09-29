# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import MethodOutputsDelivery


class TestMethodOutputsDeliver(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(TypeError, 'method_path should be'):
            _ = MethodOutputsDelivery(max_keep_data=1, method_path=1)

        with self.assertRaisesRegex(AssertionError,
                                    'method_path must have at '):
            _ = MethodOutputsDelivery(max_keep_data=1, method_path='toy_func')

        with self.assertRaisesRegex(ImportError, 'aaa is not imported'):
            _ = MethodOutputsDelivery(max_keep_data=1, method_path='aaa.bb.b')

        with self.assertRaisesRegex(AssertionError, 'bb is not in toy_module'):
            _ = MethodOutputsDelivery(
                max_keep_data=1, method_path='toy_module.bb.bbb')

        with self.assertRaisesRegex(TypeError, 'toy_func should be a type'):
            _ = MethodOutputsDelivery(
                max_keep_data=1, method_path='toy_module.toy_func.bbb')

        with self.assertRaisesRegex(AssertionError, 'bbb is not in'):
            _ = MethodOutputsDelivery(
                max_keep_data=1, method_path='toy_module.ToyClass.bbb')

        with self.assertRaisesRegex(TypeError, 'count should be'):
            _ = MethodOutputsDelivery(
                max_keep_data=1, method_path='toy_module.ToyClass.count')

    def test_context_manager(self):
        from toy_module import ToyClass

        delivery = MethodOutputsDelivery(
            max_keep_data=2, method_path='toy_module.ToyClass.random_int')

        # Without ``MethodOutputsDelivery``, outputs of the teacher and the
        # student are very likely to be different.
        # from toy_module import ToyClass
        # toy_class = ToyClass()
        # output_tea = toy_class.random_int()
        # output_stu = toy_class.random_int()

        delivery.override_data = False
        with delivery:
            toy_class = ToyClass()
            output1_tea = toy_class.random_int()
            output2_tea = toy_class.random_int()

        with self.assertRaisesRegex(AssertionError, 'push into an full queue'):
            with delivery:
                _ = toy_class.random_int()

        delivery.override_data = True
        with delivery:
            output1_stu = toy_class.random_int()
            output2_stu = toy_class.random_int()

        # With ``MethodOutputsDeliver``, outputs of the teacher and the
        # student are the same.
        assert output1_stu == output1_tea and output2_stu == output2_tea

        with self.assertRaisesRegex(AssertionError, 'pop from an empty queue'):
            with delivery:
                _ = toy_class.random_int()
