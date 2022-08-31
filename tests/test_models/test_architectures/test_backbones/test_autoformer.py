# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch

from mmrazor.models.architectures.backbones import AutoformerBackbone
from mmrazor.models.architectures.backbones.autoformer_backbone import \
    TransformerEncoderLayer
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.mutators.channel_mutator import BigNASChannelMutator
from mmrazor.models.mutators.value_mutator import (DynamicValueMutator,
                                                   ValueMutator)


class TestAutoformer(TestCase):

    def setUp(self) -> None:
        self.mutable_settings = {
            'mlp_ratios': [3.0, 3.5, 4.0],  # mutable value
            'num_heads': [8, 9, 10],  # mutable value
            'depth': [14, 15, 16],  # mutable value
            'embed_dims': [528, 576, 624],  # mutable channel
        }

    def test_init(self) -> None:
        m = AutoformerBackbone()
        i = torch.randn(8, 3, 224, 224)
        o = m(i)

        print(o[0].shape, type(o), len(o))

        assert o is not None

    def test_mutator(self):
        m = AutoformerBackbone()
        cm = BigNASChannelMutator()
        cm.prepare_from_supernet(m)
        print(cm.search_groups)
        print('=' * 10)
        vm = DynamicValueMutator()
        vm.prepare_from_supernet(m)
        print(vm.search_groups)
        print('=' * 10)
        vm2 = ValueMutator()
        vm2.prepare_from_supernet(m)
        print(vm2.search_groups)

    def test_block(self):

        self.num_head_range = self.mutable_settings['num_heads']
        self.depth_range = self.mutable_settings['depth']
        self.mlp_ratio_range = self.mutable_settings['mlp_ratios']
        self.embed_dim_range = self.mutable_settings['embed_dims']

        block = TransformerEncoderLayer(640, 10, 4, 0, 0)

        mutable_num_heads = OneShotMutableValue(
            value_list=self.num_head_range, default_value=10)
        mutable_mlp_ratios = OneShotMutableValue(
            value_list=self.mlp_ratio_range, default_value=4)
        mutable_embed_dims = OneShotMutableChannel(
            num_channels=640,
            candidate_mode='number',
            candidate_choices=self.embed_dim_range)

        block.mutate_encoder_layer(
            mutable_embed_dims=mutable_embed_dims,
            mutable_num_heads=mutable_num_heads,
            mutable_mlp_ratios=mutable_mlp_ratios)

        print('*' * 10)

        for name, module in block.named_modules():
            if isinstance(module, OneShotMutableValue):
                print(name, type(module))

        print('*' * 10)

        for name, module in block.named_modules():
            if isinstance(module, OneShotMutableChannel):
                print(name, type(module))
        print('*' * 10)


if __name__ == '__main__':
    unittest.main()
