# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import torch
import torch.nn as nn
import pytest

from mmrazor.models.architectures.dynamic_ops import DynamicRelativePosition2D
from mmrazor.models.architectures.ops import RelativePosition2D
from mmrazor.models.mutables import OneShotMutableChannel

class TestDynamicRP(TestCase):

    def setUp(self) -> None:
        mutable_head_dims = OneShotMutableChannel(
            8, candidate_choices=[2, 4, 6, 8], candidate_mode='number')

        self.dynamic_rp = DynamicRelativePosition2D(
            head_dims=8, max_relative_position=14)

        mutable_head_dims.current_choice = 6
        self.dynamic_rp.register_mutable_attr('head_dims', mutable_head_dims)

    def test_mutable_attrs(self) -> None:

        assert self.dynamic_rp.mutable_head_dims.current_choice == 6

        embed = self.dynamic_rp.forward(14, 14)

        self.assertIsNotNone(embed)

        with pytest.raises(ValueError):
            mutable = OneShotMutableChannel(
                10, candidate_choices=[6, 8, 10], candidate_mode='number')
            self.dynamic_rp.register_mutable_attr('head_dims', mutable)

    def test_convert(self):
        static_model = RelativePosition2D(
            head_dims=10, max_relative_position=14)

        dynamic_model = DynamicRelativePosition2D.convert_from(static_model)

        self.assertIsNotNone(dynamic_model)

    def test_to_static_op(self):
        with pytest.raises(RuntimeError):
            static_m = self.dynamic_rp.to_static_op()

        mutable = OneShotMutableChannel(
            8, candidate_choices=[2, 4, 6, 8], candidate_mode='number')

        mutable.current_choice = 4

        mutable.fix_chosen(mutable.dump_chosen())

        self.dynamic_rp.register_mutable_attr('head_dims', mutable)
        static_m = self.dynamic_rp.to_static_op()

        self.assertIsNotNone(static_m)

        dynamic_output = self.dynamic_rp.forward(14, 14)
        static_output =  static_m.forward(14, 14)
        assert torch.equal(dynamic_output, static_output)