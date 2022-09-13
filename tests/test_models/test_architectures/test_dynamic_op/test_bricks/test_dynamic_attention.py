# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.architectures.dynamic_ops import DynamicMultiheadAttention
from mmrazor.models.architectures.ops import MultiheadAttention
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue


class TestDynamicMHA(TestCase):

    def setUp(self) -> None:
        self.mutable_num_heads = OneShotMutableValue(
            value_list=[2, 4, 8], default_value=8)
        self.mutable_embed_dims = OneShotMutableChannel(
            num_channels=128,
            candidate_choices=[32, 64, 128],
            candidate_mode='number')
        self.mutable_q_embed_dims = self.mutable_num_heads * 8

        # derived mutable
        self.mutable_head_dims = self.mutable_q_embed_dims.derive_divide_mutable(  # noqa: E501
            self.mutable_num_heads)

        self.dynamic_m = DynamicMultiheadAttention(embed_dims=128, num_heads=8)

        self.dynamic_m.register_mutable_attr('num_heads',
                                             self.mutable_num_heads)
        self.dynamic_m.register_mutable_attr('embed_dims',
                                             self.mutable_embed_dims)
        self.dynamic_m.register_mutable_attr('q_embed_dims',
                                             self.mutable_q_embed_dims)

        self.dynamic_m.rel_pos_embed_k.register_mutable_attr(
            'head_dims', self.mutable_head_dims)
        self.dynamic_m.rel_pos_embed_v.register_mutable_attr(
            'head_dims', self.mutable_head_dims)

        self.assertEqual(
            self.dynamic_m.get_mutable_attr('num_heads').current_choice, 8)
        self.assertEqual(
            self.dynamic_m.get_mutable_attr('embed_dims').current_choice, 128)

    def test_forward(self) -> None:
        x = torch.randn(8, 197, 128)
        output = self.dynamic_m(x)
        self.assertIsNotNone(output)

    def test_convert(self) -> None:
        static_m = MultiheadAttention(embed_dims=100, num_heads=10)
        dynamic_m = DynamicMultiheadAttention.convert_from(static_m)

        self.assertIsNotNone(dynamic_m)

    def test_to_static_op(self) -> None:
        with pytest.raises(RuntimeError):
            self.dynamic_m.to_static_op()

        current_mutable_num_heads = self.dynamic_m.get_mutable_attr(
            'num_heads')
        current_mutable_embed_dims = self.dynamic_m.get_mutable_attr(
            'embed_dims')
        current_mutable_head_dims = self.dynamic_m.rel_pos_embed_k.get_mutable_attr(  # noqa: E501
            'head_dims')

        current_mutable_embed_dims.fix_chosen(
            current_mutable_embed_dims.dump_chosen())
        current_mutable_num_heads.fix_chosen(
            current_mutable_num_heads.dump_chosen())
        current_mutable_head_dims.fix_chosen(
            current_mutable_head_dims.dump_chosen())

        static_op = self.dynamic_m.to_static_op()

        self.assertIsNotNone(static_op)
