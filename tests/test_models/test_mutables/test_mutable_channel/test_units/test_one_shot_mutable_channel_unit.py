# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import OneShotMutableChannelUnit
from mmrazor.models.mutators.channel_mutator import ChannelMutator
from tests.data.models import DynamicAttention


class TestSequentialMutableChannelUnit(TestCase):

    def test_init(self):
        unit = OneShotMutableChannelUnit(
            48, [20, 30, 40], choice_mode='number', divisor=8)
        self.assertSequenceEqual(unit.candidate_choices, [24, 32, 40])

        unit = OneShotMutableChannelUnit(
            48, [0.3, 0.5, 0.7], choice_mode='ratio', divisor=8)
        self.assertSequenceEqual(unit.candidate_choices, [1 / 3, 0.5, 2 / 3])

    def test_unit_predefined(self):
        model = DynamicAttention()
        mutator = ChannelMutator(
            channel_unit_cfg={
                'type': 'OneShotMutableChannelUnit',
                'default_args': {
                    'unit_predefined': False
                }
            },
            parse_cfg={'type': 'Predefined'})
        mutator.prepare_from_supernet(model)
        self.assertSequenceEqual(mutator.units[0].candidate_choices,
                                 [576, 624])
        self.assertSequenceEqual(mutator.units[1].candidate_choices, [64])
