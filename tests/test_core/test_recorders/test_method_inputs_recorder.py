# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import MethodInputsRecorder


class TestFuncOutputsRecorder(TestCase):

    def test_context_manager(self):
        from toy_mod import ToyClass

        toy = ToyClass()

        recorder = MethodInputsRecorder('toy_mod.ToyClass.func')
        recorder.initialize()

        with recorder:
            _ = toy.func(x=1, y=2)
            _ = toy.func(1, y=2)
            _ = toy.func(y=2, x=1)

        self.assertTrue(
            recorder.get_record_data(record_idx=0, data_idx=0) == 1)
        self.assertTrue(
            recorder.get_record_data(record_idx=0, data_idx=1) == 2)

        self.assertTrue(
            recorder.get_record_data(record_idx=1, data_idx=0) == 1)
        self.assertTrue(
            recorder.get_record_data(record_idx=1, data_idx=1) == 2)

        self.assertTrue(
            recorder.get_record_data(record_idx=2, data_idx=0) == 1)
        self.assertTrue(
            recorder.get_record_data(record_idx=2, data_idx=1) == 2)
