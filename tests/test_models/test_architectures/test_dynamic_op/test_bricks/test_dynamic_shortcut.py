# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.architectures.dynamic_ops import DynamicShortcutLayer
from mmrazor.models.architectures.ops import ShortcutLayer
from mmrazor.models.mutables import OneShotMutableChannel


class TestShortcutLayer(TestCase):

    def setUp(self):
        mutable_in_features = OneShotMutableChannel(
            12, candidate_choices=[8, 10, 12])
        mutable_out_features = OneShotMutableChannel(
            12, candidate_choices=[8, 10, 12])

        conv_cfg = dict(type='mmrazor.BigNasConv2d')
        self.shortcut = DynamicShortcutLayer(
            in_channels=12, out_channels=12, conv_cfg=conv_cfg)

        self.shortcut.register_mutable_attr('in_channels', mutable_in_features)
        self.shortcut.register_mutable_attr('out_channels',
                                            mutable_out_features)

        self.assertTrue(
            self.shortcut.get_mutable_attr('in_channels').current_choice == 12)
        self.assertTrue(
            self.shortcut.get_mutable_attr('out_channels').current_choice ==
            12)

    def test_convert(self):
        static_m = ShortcutLayer(10, 8)

        dynamic_m = DynamicShortcutLayer.convert_from(static_m)

        self.assertIsNotNone(dynamic_m)

    def test_to_static_op(self):
        input = torch.randn(1, 10, 224, 224)

        mutable_in_features = OneShotMutableChannel(
            12, candidate_choices=[8, 10, 12])
        mutable_out_features = OneShotMutableChannel(
            12, candidate_choices=[8, 10, 12])

        mutable_in_features.current_choice = 10
        mutable_out_features.current_choice = 8

        with pytest.raises(RuntimeError):
            self.shortcut.to_static_op()

        mutable_in_features.fix_chosen(
            mutable_in_features.dump_chosen().chosen)
        mutable_out_features.fix_chosen(
            mutable_out_features.dump_chosen().chosen)

        self.shortcut.conv.register_mutable_attr('in_channels',
                                                 mutable_in_features)
        self.shortcut.conv.register_mutable_attr('out_channels',
                                                 mutable_out_features)

        static_op = self.shortcut.conv.to_static_op()

        x = static_op(input)
        assert x.shape[1] == mutable_out_features.current_choice
