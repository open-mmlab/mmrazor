# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrazor.models.chex.chex_unit import ChexUnit
from mmrazor.models.mutators import ChannelMutator
from ..data.models import SingleLineModel


class TestChexUnit(unittest.TestCase):

    def test_chex_unit(self):
        # test init
        model = SingleLineModel()
        mutator = ChannelMutator(channel_unit_cfg=ChexUnit)
        mutator.prepare_from_supernet(model)

        unit: ChexUnit = mutator.mutable_units[0]

        # test prune
        unit.prune(4)
        self.assertEqual(unit.current_choice, 4)

        # test bn_imp
        self.assertEqual(list(unit.bn_imp.shape), [8])
        print(model)
        print(unit.config_template(with_channels=True))

        # test grow
        unit.grow(2)
        self.assertEqual(unit.current_choice, 6)
        unit.grow(10)
        self.assertEqual(unit.current_choice, 8)
