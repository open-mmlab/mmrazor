# Copyright (c) OpenMMLab. All rights reserved.
import sys
import unittest
from typing import Union

import torch

# from mmrazor.models.mutables import MutableChannelGroup
from mmrazor.models.mutables.mutable_channel import SimpleChannelGroup
from mmrazor.models.mutators.channel_mutator import BaseChannelMutator
from mmrazor.registry import MODELS
from ...test_core.test_graph.test_graph import TestGraph

sys.setrecursionlimit(2000)


@MODELS.register_module()
class RandomChannelGroup(SimpleChannelGroup):

    def generate_mask(self, choice: Union[int, float]) -> torch.Tensor:
        if isinstance(choice, float):
            choice = max(1, int(self.num_channels * choice))
        assert 0 < choice <= self.num_channels
        rand_imp = torch.rand([self.num_channels])
        ind = rand_imp.topk(choice)[1]
        mask = torch.zeros([self.num_channels])
        mask.scatter_(-1, ind, 1)
        return mask


DATA_GROUPSS = [SimpleChannelGroup, RandomChannelGroup]


class TestChannelMutator(unittest.TestCase):

    def test_sample_subnet(self):
        data_models = TestGraph.backward_tracer_passed_models()

        for i, data in enumerate(data_models):
            with self.subTest(i=i, data=data):
                model = data()

                mutator = BaseChannelMutator(model)
                subnet = mutator.sample_subnet()
                mutator.apply_subnet(subnet)
                self.assertGreaterEqual(len(mutator.prunable_groups), 1)

                x = torch.rand([2, 3, 224, 224])
                y = model(x)
                self.assertEqual(list(y.shape), [2, 1000])

    def test_generic_support(self):
        data_models = TestGraph.backward_tracer_passed_models()

        for data_model in data_models[:1]:
            for group_type in DATA_GROUPSS:
                with self.subTest(model=data_model, unit=group_type):

                    model = data_model()

                    mutator = BaseChannelMutator(
                        model, channl_group_cfg=group_type)
                    mutator.groups

                    subnet = mutator.sample_subnet()
                    mutator.apply_subnet(subnet)

                    x = torch.rand([2, 3, 224, 224])
                    y = model(x)
                    self.assertEqual(list(y.shape), [2, 1000])
