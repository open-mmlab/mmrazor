# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase

import torch

from mmrazor.structures.graph import ModuleGraph
from tests.data.models import (AddCatModel, ConcatModel, LineModel,
                               MultiConcatModel, MultiConcatModel2, ResBlock)

sys.setrecursionlimit(int(1e8))


class ToyCNNPseudoLoss:

    def __call__(self, model):
        pseudo_img = torch.rand(2, 3, 16, 16)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


DATA = [
    {
        'model': LineModel,
        'num_nodes': 5,
    },
    {
        'model': ResBlock,
        'num_nodes': 7,
    },
    {
        'model': ConcatModel,
        'num_nodes': 7,
    },
    {
        'model': MultiConcatModel2,
        'num_nodes': 7,
    },
    {
        'model': MultiConcatModel,
        'num_nodes': 7,
    },
    {
        'model': AddCatModel
    },
]


class TestGraph(TestCase):

    def test_graph_init(self) -> None:

        for data in DATA:
            with self.subTest(data=data):
                model = data['model']()
                # print(model)
                graphs = [
                    ModuleGraph.init_using_backward_tracer(model),
                ]

                unit_num = len(graphs[0].nodes)

                for graph in graphs:

                    # check channels
                    try:
                        graph.check()
                    except Exception as e:
                        self.fail(str(e) + '\n' + str(graph))

                    # check number of nodes
                    self.assertEqual(unit_num, len(graph.nodes))
                    if 'num_nodes' in data:
                        self.assertEqual(
                            len(graph),
                            data['num_nodes'],
                            msg=f'{graph.nodes}')
