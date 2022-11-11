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

                _ = channel_graph.collect_units

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

        _ = channel_graph.collect_units


class TestChannelUnit(unittest.TestCase):

    def test_union(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        channel_tensor3 = ChannelTensor(8)
        channel_tensor4 = ChannelTensor(8)
        unit1 = channel_tensor1.unit_dict[(0, 8)]
        unit2 = channel_tensor2.unit_dict[(0, 8)]
        unit3 = channel_tensor3.unit_dict[(0, 8)]
        unit4 = channel_tensor4.unit_dict[(0, 8)]

        unit12 = BaseChannelUnit.union_two_units(unit1, unit2)
        self.assertDictEqual(channel_tensor1.unit_dict,
                             channel_tensor2.unit_dict)

        unit34 = BaseChannelUnit.union_two_units(unit3, unit4)
        BaseChannelUnit.union_two_units(unit12, unit34)
        self.assertDictEqual(channel_tensor1.unit_dict,
                             channel_tensor4.unit_dict)

    def test_split(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        BaseChannelUnit.union_two_units(channel_tensor1.unit_dict[(0, 8)],
                                        channel_tensor2.unit_dict[(0, 8)])
        unit1 = channel_tensor1.unit_dict[(0, 8)]
        BaseChannelUnit.split_unit(unit1, [2, 6])

        self.assertDictEqual(channel_tensor1.unit_dict,
                             channel_tensor2.unit_dict)


class TestChannelTensor(unittest.TestCase):

    def test_init(self):
        channel_tensor = ChannelTensor(8)
        self.assertIn((0, 8), channel_tensor.unit_dict)

    def test_align_with_nums(self):
        channel_tensor = ChannelTensor(8)
        channel_tensor.align_units_with_nums([2, 6])
        self.assertSequenceEqual(
            list(channel_tensor.unit_dict.keys()), [(0, 2), (2, 8)])
        channel_tensor.align_units_with_nums([2, 2, 4])
        self.assertSequenceEqual(
            list(channel_tensor.unit_dict.keys()), [(0, 2), (2, 4), (4, 8)])

    def test_align_units(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        channel_tensor3 = ChannelTensor(8)

        BaseChannelUnit.split_unit(channel_tensor1.unit_list[0], [2, 6])
        BaseChannelUnit.split_unit(channel_tensor2.unit_list[0], [4, 4])
        BaseChannelUnit.split_unit(channel_tensor3.unit_list[0], [6, 2])
        """
        xxoooooo
        xxxxoooo
        xxxxxxoo
        """

        ChannelTensor.align_tensors(channel_tensor1, channel_tensor2,
                                    channel_tensor3)
        for lst in [channel_tensor1, channel_tensor2, channel_tensor3]:
            self.assertSequenceEqual(
                list(lst.unit_dict.keys()), [
                    (0, 2),
                    (2, 4),
                    (4, 6),
                    (6, 8),
                ])

    def test_expand(self):
        channel_tensor = ChannelTensor(8)
        expanded = channel_tensor.expand(4)
        self.assertIn((0, 32), expanded.unit_dict)

    def test_union(self):
        channel_tensor1 = ChannelTensor(8)
        channel_tensor2 = ChannelTensor(8)
        channel_tensor3 = ChannelTensor(8)
        channel_tensor4 = ChannelTensor(8)
        channel_tensor3.union(channel_tensor4)

        self.assertEqual(
            id(channel_tensor3.unit_dict[(0, 8)]),
            id(channel_tensor4.unit_dict[(0, 8)]))

        channel_tensor2.union(channel_tensor3)
        channel_tensor1.union(channel_tensor2)

        self.assertEqual(
            id(channel_tensor1.unit_dict[(0, 8)]),
            id(channel_tensor2.unit_dict[(0, 8)]))
        self.assertEqual(
            id(channel_tensor2.unit_dict[(0, 8)]),
            id(channel_tensor3.unit_dict[(0, 8)]))
        self.assertEqual(
            id(channel_tensor3.unit_dict[(0, 8)]),
            id(channel_tensor4.unit_dict[(0, 8)]))
