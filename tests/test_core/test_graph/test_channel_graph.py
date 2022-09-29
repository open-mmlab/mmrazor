# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from torch import nn

from mmrazor.models.task_modules import BackwardTracer
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import ModuleGraph
from mmrazor.structures.graph.channel_graph import ChannelGraph
from mmrazor.structures.graph.channel_modules import (BaseChannelUnit,
                                                      ChannelTensor)
from mmrazor.structures.graph.channel_nodes import \
    default_channel_node_converter
from ...data.models import LineModel
from .test_graph import TestGraph

NodeMap = {}


@TASK_UTILS.register_module()
class ImageClassifierPseudoLossWithSixChannel:
    """Calculate the pseudo loss to trace the topology of a `ImageClassifier`
    in MMClassification with `BackwardTracer`."""

    def __call__(self, model) -> torch.Tensor:
        pseudo_img = torch.rand(1, 6, 224, 224)
        pseudo_output = model(pseudo_img)
        return sum(pseudo_output)


class TestChannelGraph(unittest.TestCase):

    def test_init(self):
        model = LineModel()
        module_graph = ModuleGraph.init_from_backward_tracer(model)

        _ = ChannelGraph.copy_from(module_graph,
                                   default_channel_node_converter)

    def test_forward(self):
        for model_data in TestGraph.backward_tracer_passed_models():
            with self.subTest(model=model_data):
                model = model_data()
                module_graph = ModuleGraph.init_from_backward_tracer(model)

                channel_graph = ChannelGraph.copy_from(
                    module_graph, default_channel_node_converter)
                channel_graph.forward()

                _ = channel_graph.collect_groups

    def test_forward_with_config_num_in_channel(self):

        class MyModel(nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(6, 3, 3, 1, 1)
                self.net = LineModel()

            def forward(self, x):
                return self.net(self.conv1(x))

        model = MyModel()
        module_graph = ModuleGraph.init_from_backward_tracer(
            model,
            backward_tracer=BackwardTracer(
                loss_calculator=ImageClassifierPseudoLossWithSixChannel()))

        channel_graph = ChannelGraph.copy_from(module_graph,
                                               default_channel_node_converter)
        channel_graph.forward(num_input_channel=6)

        _ = channel_graph.collect_groups


class TestChannelUnit(unittest.TestCase):

    def test_union(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        channel_tensor3 = ChannelTensor(8)
        channel_tensor4 = ChannelTensor(8)
        group1 = channel_tensor1.group_dict[(0, 8)]
        group2 = channel_tensor2.group_dict[(0, 8)]
        group3 = channel_tensor3.group_dict[(0, 8)]
        group4 = channel_tensor4.group_dict[(0, 8)]

        group12 = BaseChannelUnit.union_two_groups(group1, group2)
        self.assertDictEqual(channel_tensor1.group_dict,
                             channel_tensor2.group_dict)

        group34 = BaseChannelUnit.union_two_groups(group3, group4)
        BaseChannelUnit.union_two_groups(group12, group34)
        self.assertDictEqual(channel_tensor1.group_dict,
                             channel_tensor4.group_dict)

    def test_split(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        BaseChannelUnit.union_two_groups(channel_tensor1.group_dict[(0, 8)],
                                          channel_tensor2.group_dict[(0, 8)])
        group1 = channel_tensor1.group_dict[(0, 8)]
        BaseChannelUnit.split_group(group1, [2, 6])

        self.assertDictEqual(channel_tensor1.group_dict,
                             channel_tensor2.group_dict)


class TestChannelTensor(unittest.TestCase):

    def test_init(self):
        channel_tensor = ChannelTensor(8)
        self.assertIn((0, 8), channel_tensor.group_dict)

    def test_align_with_nums(self):
        channel_tensor = ChannelTensor(8)
        channel_tensor.align_groups_with_nums([2, 6])
        self.assertSequenceEqual(
            list(channel_tensor.group_dict.keys()), [(0, 2), (2, 8)])
        channel_tensor.align_groups_with_nums([2, 2, 4])
        self.assertSequenceEqual(
            list(channel_tensor.group_dict.keys()), [(0, 2), (2, 4), (4, 8)])

    def test_align_groups(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        channel_tensor3 = ChannelTensor(8)

        BaseChannelUnit.split_group(channel_tensor1.group_list[0], [2, 6])
        BaseChannelUnit.split_group(channel_tensor2.group_list[0], [4, 4])
        BaseChannelUnit.split_group(channel_tensor3.group_list[0], [6, 2])
        """
        xxoooooo
        xxxxoooo
        xxxxxxoo
        """

        ChannelTensor.align_tensors(channel_tensor1, channel_tensor2,
                                    channel_tensor3)
        for lst in [channel_tensor1, channel_tensor2, channel_tensor3]:
            self.assertSequenceEqual(
                list(lst.group_dict.keys()), [
                    (0, 2),
                    (2, 4),
                    (4, 6),
                    (6, 8),
                ])

    def test_expand(self):
        channel_tensor = ChannelTensor(8)
        expanded = channel_tensor.expand(4)
        self.assertIn((0, 32), expanded.group_dict)

    def test_union(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        channel_tensor3 = ChannelTensor(8)
        channel_tensor4 = ChannelTensor(8)
        channel_tensor3.union(channel_tensor4)

        self.assertEqual(
            id(channel_tensor3.group_dict[(0, 8)]),
            id(channel_tensor4.group_dict[(0, 8)]))

        channel_tensor2.union(channel_tensor3)
        channel_tensor1.union(channel_tensor2)

        self.assertEqual(
            id(channel_tensor1.group_dict[(0, 8)]),
            id(channel_tensor2.group_dict[(0, 8)]))
        self.assertEqual(
            id(channel_tensor2.group_dict[(0, 8)]),
            id(channel_tensor3.group_dict[(0, 8)]))
        self.assertEqual(
            id(channel_tensor3.group_dict[(0, 8)]),
            id(channel_tensor4.group_dict[(0, 8)]))
