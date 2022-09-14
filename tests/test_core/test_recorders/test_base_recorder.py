# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from toy_mod import Toy

from mmrazor.models.task_modules import MethodOutputsRecorder


class TestFuncOutputsRecorder(TestCase):

    def test_get_record_data(self):

        toy = Toy()

        recorder = MethodOutputsRecorder('toy_mod.Toy.toy_func')
        recorder.initialize()

        with recorder:
            res0 = toy.toy_func()
            res1 = toy.toy_func()

        self.assertEquals(res0, recorder.get_record_data(record_idx=0))
        self.assertEquals(res1, recorder.get_record_data(record_idx=1))

        with self.assertRaisesRegex(
                AssertionError,
                'record_idx is illegal. The length of data_buffer is 2, '
                'but record_idx is 2'):
            _ = recorder.get_record_data(record_idx=2)

        with self.assertRaisesRegex(
                TypeError,
                'When data_idx is not None, record should be a list or '
                'tuple instance'):
            _ = recorder.get_record_data(data_idx=0)

        recorder = MethodOutputsRecorder('toy_mod.Toy.toy_list_func')
        recorder.initialize()

        with recorder:
            res = toy.toy_list_func()

        self.assertEqual(len(res), 3)

        with self.assertRaisesRegex(
                AssertionError,
                'data_idx is illegal. The length of record is 3'):
            _ = recorder.get_record_data(data_idx=3)

        self.assertEquals(res[2], recorder.get_record_data(data_idx=2))
