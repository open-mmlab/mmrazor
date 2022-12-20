# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmrazor.models.chex.chex_mutator import ChexMutator
from mmrazor.models.chex.chex_unit import ChexUnit
from ..data.models import SingleLineModel


class TestChexMutator(unittest.TestCase):

    def test_chex_mutator(self):
        model = SingleLineModel()
        mutator = ChexMutator(channel_unit_cfg=ChexUnit)
        mutator.prepare_from_supernet(model)

        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.weight.data = torch.rand_like(module.weight.data)

        mutator.prune()
        print(mutator.current_choices)

        mutator.grow(0.2)
        print(mutator.current_choices)
