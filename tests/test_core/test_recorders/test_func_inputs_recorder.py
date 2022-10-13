# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.task_modules import FunctionInputsRecorder


class TestFuncOutputsRecorder(TestCase):

    def test_context_manager(self):
        from toy_mod import execute_toy_func2 as execute_toy_func

        recorder = FunctionInputsRecorder('toy_mod.toy_func2')
        recorder.initialize()

        with recorder:
            execute_toy_func(1, 2)
            execute_toy_func(1, b=2)
            execute_toy_func(b=2, a=1)

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
