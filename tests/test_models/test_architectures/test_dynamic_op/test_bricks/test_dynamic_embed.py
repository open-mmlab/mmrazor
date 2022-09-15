# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from mmcls.models.utils import PatchEmbed

from mmrazor.models.architectures.dynamic_ops import DynamicPatchEmbed
from mmrazor.models.mutables import OneShotMutableChannel


class TestPatchEmbed(TestCase):

    def setUp(self):
        self.dynamic_embed = DynamicPatchEmbed(
            img_size=224, in_channels=3, embed_dims=100)

        mutable_embed_dims = OneShotMutableChannel(
            100, candidate_choices=[10, 50, 100], candidate_mode='number')

        mutable_embed_dims.current_choice = 50
        self.dynamic_embed.register_mutable_attr('embed_dims',
                                                 mutable_embed_dims)

    def test_patch_embed(self):
        mutable = OneShotMutableChannel(
            120, candidate_choices=[10, 50, 120], candidate_mode='number')

        with pytest.raises(ValueError):
            self.dynamic_embed.register_mutable_attr('embed_dims', mutable)

        self.assertTrue(
            self.dynamic_embed.get_mutable_attr('embed_dims').current_choice ==
            50)

    def test_convert(self):
        static_m = PatchEmbed(img_size=224, in_channels=3, embed_dims=768)

        dynamic_m = DynamicPatchEmbed.convert_from(static_m)

        self.assertIsNotNone(dynamic_m)

    def test_to_static_op(self):
        mutable_embed_dims = OneShotMutableChannel(
            100, candidate_choices=[10, 50, 100], candidate_mode='number')
        mutable_embed_dims.current_choice = 10

        with pytest.raises(RuntimeError):
            self.dynamic_embed.to_static_op()

        mutable_embed_dims.fix_chosen(mutable_embed_dims.dump_chosen())
        self.dynamic_embed.register_mutable_attr('embed_dims',
                                                 mutable_embed_dims)
        static_op = self.dynamic_embed.to_static_op()

        self.assertIsNotNone(static_op)

        x = torch.randn(8, 3, 224, 224)
        dynamic_output = self.dynamic_embed.forward(x)
        static_output = static_op.forward(x)
        self.assertTrue(torch.equal(dynamic_output, static_output))
