# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
from torch.nn import LayerNorm

from mmrazor.models.architectures.dynamic_ops import DynamicLayerNorm
from mmrazor.models.mutables import SquentialMutableChannel


class TestDynamicLayerNorm(TestCase):

    def setUp(self) -> None:
        self.dynamic_m = DynamicLayerNorm(100)

        mutable_num_features = SquentialMutableChannel(num_channels=100)

        mutable_num_features.current_choice = 50

        self.dynamic_m.register_mutable_attr('num_features',
                                             mutable_num_features)

    def test_init(self) -> None:
        mutable = SquentialMutableChannel(num_channels=100)
        self.dynamic_m.register_mutable_attr('in_channels', mutable)
        self.dynamic_m.register_mutable_attr('out_channels', mutable)

        self.assertEqual(
            self.dynamic_m.get_mutable_attr('num_features').current_choice, 50)

    def test_to_static_op(self):
        with pytest.raises(RuntimeError):
            self.dynamic_m.to_static_op()

        current_mutable = self.dynamic_m.get_mutable_attr('num_features')
        current_mutable.fix_chosen(current_mutable.dump_chosen().chosen)
        static_op = self.dynamic_m.to_static_op()

        self.assertIsNotNone(static_op)

    def test_convert(self) -> None:
        static_m = LayerNorm(100)
        dynamic_m = DynamicLayerNorm.convert_from(static_m)

        self.assertIsNotNone(dynamic_m)
