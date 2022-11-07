# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutators.channel_mutator import ChannelMutator
from .....data.models import DynamicAttention


class TestOneShotMutableChannelUnit_VIT(TestCase):

    def test_init(self):
        model = DynamicAttention()
        mutator = ChannelMutator(
            channel_unit_cfg={
                'type': 'OneShotMutableChannelUnit_VIT',
                'default_args': {}
            },
            parse_cfg={'type': 'Predefined'})
        mutator.prepare_from_supernet(model)
        choices = mutator.sample_choices()
        mutator.set_choices(choices)
        self.assertSequenceEqual(mutator.units[0].candidate_choices,
                                 [576, 624])
        self.assertSequenceEqual(mutator.units[1].candidate_choices, [64])
