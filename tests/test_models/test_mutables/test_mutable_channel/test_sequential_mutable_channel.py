# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models.mutables import SquentialMutableChannel


class TestSquentialMutableChannel(TestCase):

    def _test_mutable(self,
                      mutable: SquentialMutableChannel,
                      set_choice,
                      get_choice,
                      activate_channels,
                      mask=None):
        mutable.current_choice = set_choice
        assert mutable.current_choice == get_choice
        assert mutable.activated_channels == activate_channels
        if mask is not None:
            assert (mutable.current_mask == mask).all()

    def _generate_mask(self, num: int, all: int):
        mask = torch.zeros([all])
        mask[0:num] = 1
        return mask.bool()

    def test_mul_float(self):
        channel = SquentialMutableChannel(10)
        new_channel = channel * 0.5
        self.assertEqual(new_channel.current_choice, 5)
        channel.current_choice = 5
        self.assertEqual(new_channel.current_choice, 2)

    def test_int_choice(self):
        channel = SquentialMutableChannel(10)
        self._test_mutable(channel, 5, 5, 5, self._generate_mask(5, 10))
        self._test_mutable(channel, 0.2, 2, 2, self._generate_mask(2, 10))

    def test_float_choice(self):
        channel = SquentialMutableChannel(10, choice_mode='ratio')
        self._test_mutable(channel, 0.5, 0.5, 5, self._generate_mask(5, 10))
        self._test_mutable(channel, 2, 0.2, 2, self._generate_mask(2, 10))
