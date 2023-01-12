# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.mutables import (SimpleMutableChannel,
                                     SquentialMutableChannel)


class TestMutableChannels(unittest.TestCase):

    def test_SquentialMutableChannel(self):
        mutable_channel = SquentialMutableChannel(4)
        mutable_channel.current_choice = 3
        self.assertEqual(mutable_channel.activated_channels,
                         mutable_channel.current_choice)
        self.assertTrue(
            (mutable_channel.current_mask == torch.tensor([1, 1, 1,
                                                           0]).bool()).all())
        channel_str = mutable_channel.__repr__()
        self.assertEqual(
            channel_str,
            'SquentialMutableChannel(num_channels=4, activated_channels=3)')

        mutable_channel.fix_chosen()
        mutable_channel.dump_chosen()

    def test_SimpleMutableChannel(self):
        channel = SimpleMutableChannel(4)
        channel.current_choice = torch.tensor([1, 0, 0, 0]).bool()
        self.assertEqual(channel.activated_channels, 1)
        channel.fix_chosen()
        channel.dump_chosen()
