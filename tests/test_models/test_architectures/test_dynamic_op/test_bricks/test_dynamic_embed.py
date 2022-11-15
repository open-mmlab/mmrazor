# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
from mmcls.models.utils import PatchEmbed

from mmrazor.models.architectures.dynamic_ops import DynamicPatchEmbed
from mmrazor.models.mutables import SquentialMutableChannel


class TestPatchEmbed(TestCase):

    def setUp(self):
        self.dynamic_embed = DynamicPatchEmbed(
            img_size=224, in_channels=3, embed_dims=100)

        mutable_embed_dims = SquentialMutableChannel(num_channels=100)

        mutable_embed_dims.current_choice = 50
        self.dynamic_embed.register_mutable_attr('embed_dims',
                                                 mutable_embed_dims)

    def test_patch_embed(self):
        mutable = SquentialMutableChannel(num_channels=120)

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
        mutable_embed_dims = SquentialMutableChannel(num_channels=100)

        mutable_embed_dims.current_choice = 10

        with pytest.raises(RuntimeError):
            self.dynamic_embed.to_static_op()

        mutable_embed_dims.fix_chosen(mutable_embed_dims.dump_chosen().chosen)
        self.dynamic_embed.register_mutable_attr('embed_dims',
                                                 mutable_embed_dims)
        static_op = self.dynamic_embed.to_static_op()

        self.assertIsNotNone(static_op)
