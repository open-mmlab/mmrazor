# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_ops import DynamicMultiheadAttention
from mmrazor.models.architectures.ops import MultiheadAttention
from mmrazor.models.mutables import (MutableChannelContainer,
                                     OneShotMutableChannel,
                                     OneShotMutableChannelUnit,
                                     OneShotMutableValue)


class TestDynamicMHA(TestCase):

    def setUp(self) -> None:
        self.mutable_num_heads = OneShotMutableValue(
            value_list=[2, 4, 8], default_value=8)
        self.mutable_embed_dims = OneShotMutableChannel(num_channels=128)
        self.base_embed_dims = OneShotMutableChannel(
            num_channels=8, candidate_choices=[8])
        self.mutable_q_embed_dims = self.mutable_num_heads * \
            self.base_embed_dims

        self.dynamic_m = DynamicMultiheadAttention(embed_dims=128, num_heads=8)

        OneShotMutableChannelUnit._register_channel_container(
            self.dynamic_m, MutableChannelContainer)

        self.dynamic_m.register_mutable_attr('num_heads',
                                             self.mutable_num_heads)

        MutableChannelContainer.register_mutable_channel_to_module(
            self.dynamic_m, self.mutable_embed_dims, False)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.dynamic_m, self.mutable_q_embed_dims, True, end=64)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.dynamic_m.rel_pos_embed_k, self.base_embed_dims, False)
        MutableChannelContainer.register_mutable_channel_to_module(
            self.dynamic_m.rel_pos_embed_v, self.base_embed_dims, False)

    def test_forward(self) -> None:
        x = torch.randn(8, 197, 128)
        output = self.dynamic_m(x)
        self.assertIsNotNone(output)

    def test_convert(self) -> None:
        static_m = MultiheadAttention(embed_dims=100, num_heads=10)
        dynamic_m = DynamicMultiheadAttention.convert_from(static_m)
        self.assertIsNotNone(dynamic_m)
