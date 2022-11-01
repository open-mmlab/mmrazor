# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch.nn as nn
from torch.nn import Sequential

from mmrazor.models.architectures.dynamic_ops import DynamicSequential
from mmrazor.models.mutables import OneShotMutableValue


class TestDynamicSequential(TestCase):

    def setUp(self) -> None:
        self.layers = [
            nn.Linear(4, 5),
            nn.Linear(5, 6),
            nn.Linear(6, 7),
            nn.Linear(7, 8),
        ]
        self.dynamic_m = DynamicSequential(*self.layers)
        mutable_depth = OneShotMutableValue(
            value_list=[2, 3, 4], default_value=3)

        self.dynamic_m.register_mutable_attr('depth', mutable_depth)

    def test_init(self) -> None:
        self.assertEqual(
            self.dynamic_m.get_mutable_attr('depth').current_choice, 3)

    def test_to_static_op(self) -> None:
        with pytest.raises(RuntimeError):
            self.dynamic_m.to_static_op()

        current_mutable = self.dynamic_m.get_mutable_attr('depth')
        current_mutable.fix_chosen(current_mutable.dump_chosen().chosen)

        static_op = self.dynamic_m.to_static_op()
        self.assertIsNotNone(static_op)

    def test_convert_from(self) -> None:
        static_m = Sequential(*self.layers)

        dynamic_m = DynamicSequential.convert_from(static_m)

        self.assertIsNotNone(dynamic_m)
