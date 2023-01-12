# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from torch import nn

from mmrazor.models.task_modules import BackwardTracer
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import ModuleGraph
from mmrazor.structures.graph.channel_graph import ChannelGraph
from mmrazor.structures.graph.channel_nodes import \
    default_channel_node_converter
from ...data.models import SingleLineModel

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
        model = SingleLineModel()
        module_graph = ModuleGraph.init_from_backward_tracer(model)

        _ = ChannelGraph.copy_from(module_graph,
                                   default_channel_node_converter)

    # def test_forward(self):
    #     for model_data in BackwardPassedModelManager.include_models(  # noqa
    #     ):  # noqa
    #         with self.subTest(model=model_data):
    #             model = model_data()
    #             module_graph = ModuleGraph.init_from_backward_tracer(model)

    #             channel_graph = ChannelGraph.copy_from(
    #                 module_graph, default_channel_node_converter)
    #             channel_graph.forward()

    #             # units = channel_graph.collect_units()
    #             _ = channel_graph.generate_units_config()

    def test_forward_with_config_num_in_channel(self):

        class MyModel(nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(6, 3, 3, 1, 1)
                self.net = SingleLineModel()

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

        _ = channel_graph.generate_units_config
