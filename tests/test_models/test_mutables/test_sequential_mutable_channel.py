# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import SquentialMutableChannel


class TestSquentialMutableChannel(TestCase):

    def test_mul_float(self):
        channel = SquentialMutableChannel(10)
        new_channel = channel * 0.5
        self.assertEqual(new_channel.current_choice, 5)
        channel.current_choice = 5
        self.assertEqual(new_channel.current_choice, 2)
