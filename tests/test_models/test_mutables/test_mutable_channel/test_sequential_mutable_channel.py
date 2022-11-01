# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models.mutables import (OneShotMutableValue,
                                     SquentialMutableChannel)


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

    def test_mutable_channel_mul(self):
        channel = SquentialMutableChannel(2)
        self.assertEqual(channel.current_choice, 2)
        mv = OneShotMutableValue(value_list=[1, 2, 3], default_value=3)
        derived1 = channel * mv
        derived2 = mv * channel
        assert derived1.current_choice == 6
        assert derived2.current_choice == 6
        mv.current_choice = mv.min_choice
        assert derived1.current_choice == 2
        assert derived2.current_choice == 2
        assert torch.equal(derived1.current_mask, derived2.current_mask)
