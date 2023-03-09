# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Union
from unittest import TestCase

import torch

from mmrazor import digit_version
from mmrazor.implementations.pruning.group_fisher.prune_sub_model import \
    GroupFisherSubModel
from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import ChannelMutator
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


def get_model_structure(model):
    algorithm = PruneAlgorithm(copy.deepcopy(model))
    return algorithm.mutator.current_choices


class TestPruneSubModel(TestCase):

    def check_torch_version(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')

    def test_build_sub_model(self):
        self.check_torch_version()
        x = torch.rand([1, 3, 224, 224])
        model = MMClsResNet18()
        algorithm = PruneAlgorithm(model)
        algorithm.random_prune()

        # test divisor
        static_model1 = GroupFisherSubModel(algorithm, divisor=1)
        self.assertSequenceEqual(
            list(algorithm.mutator.current_choices.values()),
            list(get_model_structure(static_model1).values()))

        static_model2 = GroupFisherSubModel(algorithm, divisor=8)
        for value in get_model_structure(static_model2).values():
            self.assertTrue(value % 8 == 0)

        y1 = static_model1(x)
        y2 = static_model2(x)
        self.assertTrue((y1 - y2).abs().max() < 1e-3)
