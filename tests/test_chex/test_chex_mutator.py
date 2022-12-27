# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmrazor.models.chex.chex_mutator import ChexMutator
from mmrazor.models.chex.chex_unit import ChexUnit
from ..data.models import SingleLineModel


class TestChexMutator(unittest.TestCase):

    def test_chex_mutator(self):

        def sum_choices(choices):
            num = 0
            for i in choices.values():
                num += i
            return num

        model = SingleLineModel()
        mutator = ChexMutator(channel_unit_cfg=ChexUnit, channel_ratio=0.5)
        mutator.prepare_from_supernet(model)
        full_choices = mutator.current_choices

        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.weight.data = torch.rand_like(module.weight.data)

        mutator.prune()
        print(mutator.current_choices)
        self.assertEqual(
            sum_choices(mutator.current_choices),
            int(0.5 * sum_choices(full_choices)),
        )
        mutator.grow(0.5)
        print(mutator.current_choices)
        self.assertDictEqual(mutator.current_choices, full_choices)
