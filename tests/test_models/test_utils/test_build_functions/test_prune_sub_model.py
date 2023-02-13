# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union
from unittest import TestCase

import torch

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import ChannelMutator
from mmrazor.models.utils.build_functions import PruneSubModel
from mmrazor.registry import MODELS
from ....data.models import MMClsResNet18


class PruneAlgorithm(BaseAlgorithm):

    def __init__(self,
                 architecture,
                 mutator: Union[Dict, ChannelMutator] = dict(
                     type='ChannelMutator',
                     channel_unit_cfg=dict(
                         type='SequentialMutableChannelUnit')),
                 data_preprocessor=None,
                 init_cfg=None) -> None:
        super().__init__(
            architecture, data_preprocessor, init_cfg, module_inplace=False)
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        assert isinstance(mutator, ChannelMutator)
        self.mutator = mutator
        mutator.prepare_from_supernet(self.architecture)

    def random_prune(self):
        choices = self.mutator.sample_choices()
        self.mutator.set_choices(choices)


class TestPruneSubModel(TestCase):

    def test_init(self):
        x = torch.rand([1, 3, 224, 224])
        model = MMClsResNet18()
        algorithm = PruneAlgorithm(model)
        algorithm.random_prune()

        static_model1 = PruneSubModel(algorithm, divisor=1)
        print(static_model1)

        static_model2 = PruneSubModel(algorithm, divisor=8)
        print(static_model2)

        y1 = static_model1(x)
        y2 = static_model2(x)
        self.assertTrue((y1 - y2).abs().max() < 1e-3)
