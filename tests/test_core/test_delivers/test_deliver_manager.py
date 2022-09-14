# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmengine import ConfigDict

from mmrazor.models.task_modules import DistillDeliveryManager


class TestDeliverManager(TestCase):

    def test_init(self):

        distill_deliveries = ConfigDict(
            delivery1=dict(
                type='MethodOutputs',
                max_keep_data=2,
                method_path='toy_module.ToyClass.random_int'))

        manager = DistillDeliveryManager(distill_deliveries)
        self.assertEquals(len(manager.deliveries), 1)

        manager = DistillDeliveryManager()
        self.assertEquals(len(manager.deliveries), 0)

    def test_context_manager(self):
        from toy_module import ToyClass

        distill_deliveries = ConfigDict(
            delivery1=dict(
                type='MethodOutputs',
                max_keep_data=2,
                method_path='toy_module.ToyClass.random_int'))

        manager = DistillDeliveryManager(distill_deliveries)

        manager.override_data = False
        self.assertFalse(manager.override_data)
        with manager:
            toy_class = ToyClass()
            output1_tea = toy_class.random_int()
            output2_tea = toy_class.random_int()

        with self.assertRaisesRegex(AssertionError, 'push into an full queue'):
            with manager:
                _ = toy_class.random_int()

        self.assertFalse(manager.override_data)
        manager.override_data = True
        self.assertTrue(manager.override_data)
        with manager:
            output1_stu = toy_class.random_int()
            output2_stu = toy_class.random_int()

        # With ``DistillDeliverManager``, outputs of the teacher and
        # the student are the same.
        assert output1_stu == output1_tea and output2_stu == output2_tea

        with self.assertRaisesRegex(AssertionError, 'pop from an empty queue'):
            with manager:
                _ = toy_class.random_int()
