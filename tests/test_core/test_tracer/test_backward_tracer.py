# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from torch import Tensor, nn
from torch.nn import Module

from mmrazor.models.task_modules import (BackwardTracer, Path, PathConcatNode,
                                         PathConvNode, PathDepthWiseConvNode,
                                         PathLinearNode, PathList,
                                         PathNormNode)

NONPASS_NODES = (PathConvNode, PathLinearNode, PathConcatNode)
PASS_NODES = (PathNormNode, PathDepthWiseConvNode)


class MultiConcatModel(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(16, 8, 1)
        self.op4 = nn.Conv2d(3, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.op3(cat1)
        x4 = self.op4(x)
        output = torch.cat([x3, x4], dim=1)

        return output


class MultiConcatModel2(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(3, 8, 1)
        self.op4 = nn.Conv2d(24, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        x3 = self.op3(x)
        cat1 = torch.cat([x1, x2], dim=1)
        cat2 = torch.cat([cat1, x3], dim=1)
        output = self.op4(cat2)

        return output


class MultiConcatModel3(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(3, 8, 1)
        self.op4 = nn.Conv2d(24, 8, 1)
        self.op5 = nn.Conv2d(24, 8, 1)
        self.op6 = nn.Conv2d(24, 8, 1)
        self.op7 = nn.Conv2d(24, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        x3 = self.op3(x)
        cat1 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.op4(cat1)
        x5 = self.op5(cat1)
        x6 = self.op6(cat1)
        x7 = self.op7(cat1)
        return torch.cat([x4, x5, x6, x7], dim=1)


class ResBlock(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(8, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.op3 = nn.Conv2d(8, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bn1(self.op1(x))
        x2 = self.bn2(self.op2(x1))
        x3 = self.op3(x2 + x1)
        return x3


class ToyCNNPseudoLoss:

    def __init__(self, input_shape=(2, 3, 16, 16)):
        self.input_shape = input_shape

    def __call__(self, model):
        pseudo_img = torch.rand(self.input_shape)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


class TestBackwardTracer(TestCase):

    def test_trace_resblock(self) -> None:
        model = ResBlock()
        loss_calculator = ToyCNNPseudoLoss()
        tracer = BackwardTracer(loss_calculator=loss_calculator)
        path_list = tracer.trace(model)

        # test tracer and parser
        assert len(path_list) == 2
        assert len(path_list[0]) == 5

        # test path_list
        nonpass2parents = path_list.find_nodes_parents(NONPASS_NODES)
        assert len(nonpass2parents) == 3
        assert nonpass2parents['op1'] == list()
        assert nonpass2parents['op2'] == list({PathNormNode('bn1')})
        assert nonpass2parents['op3'] == list(
            {PathNormNode('bn2'), PathNormNode('bn1')})

        nonpass2nonpassparents = path_list.find_nodes_parents(
            NONPASS_NODES, non_pass=NONPASS_NODES)
        assert len(nonpass2parents) == 3
        assert nonpass2nonpassparents['op1'] == list()
        assert nonpass2nonpassparents['op2'] == list({PathConvNode('op1')})
        assert nonpass2nonpassparents['op3'] == list(
            {PathConvNode('op2'), PathConvNode('op1')})

        pass2nonpassparents = path_list.find_nodes_parents(
            PASS_NODES, non_pass=NONPASS_NODES)
        assert len(pass2nonpassparents) == 2
        assert pass2nonpassparents['bn1'] == list({PathConvNode('op1')})
        assert pass2nonpassparents['bn2'] == list({PathConvNode('op2')})

    def test_trace_multi_cat(self) -> None:
        loss_calculator = ToyCNNPseudoLoss()

        model = MultiConcatModel()
        tracer = BackwardTracer(loss_calculator=loss_calculator)
        path_list = tracer.trace(model)

        assert len(path_list) == 1

        nonpass2parents = path_list.find_nodes_parents(NONPASS_NODES)
        assert len(nonpass2parents) == 4
        assert nonpass2parents['op1'] == list()
        assert nonpass2parents['op2'] == list()
        path_list1 = PathList(Path(PathConvNode('op1')))
        path_list2 = PathList(Path(PathConvNode('op2')))
        # only one parent
        assert len(nonpass2parents['op3']) == 1
        assert isinstance(nonpass2parents['op3'][0], PathConcatNode)
        assert len(nonpass2parents['op3'][0]) == 2
        assert nonpass2parents['op3'][0].get_module_names() == ['op1', 'op2']
        assert nonpass2parents['op3'][0].path_lists == [path_list1, path_list2]
        assert nonpass2parents['op3'][0][0] == path_list1
        assert nonpass2parents['op4'] == list()

        model = MultiConcatModel2()
        tracer = BackwardTracer(loss_calculator=loss_calculator)
        path_list = tracer.trace(model)
        assert len(path_list) == 1

        nonpass2parents = path_list.find_nodes_parents(NONPASS_NODES)
        assert len(nonpass2parents) == 4
        assert nonpass2parents['op1'] == list()
        assert nonpass2parents['op2'] == list()
        assert nonpass2parents['op3'] == list()
        # only one parent
        assert len(nonpass2parents['op4']) == 1
        assert isinstance(nonpass2parents['op4'][0], PathConcatNode)
        assert nonpass2parents['op4'][0].get_module_names() == [
            'op1', 'op2', 'op3'
        ]

        model = MultiConcatModel3()
        tracer = BackwardTracer(loss_calculator=loss_calculator)
        path_list = tracer.trace(model)
        assert len(path_list) == 1

        nonpass2parents = path_list.find_nodes_parents(NONPASS_NODES)
        assert nonpass2parents['op1'] == list()
        assert nonpass2parents['op2'] == list()
        assert nonpass2parents['op3'] == list()
        assert nonpass2parents['op4'] == nonpass2parents['op5'] == \
               nonpass2parents['op6'] == nonpass2parents['op7']

    def test_repr(self):
        toy_node = PathConvNode('op1')
        assert repr(toy_node) == 'PathConvNode(\'op1\')'

        toy_path = Path([PathConvNode('op1'), PathConvNode('op2')])
        assert repr(
            toy_path
        ) == 'Path(\n  PathConvNode(\'op1\'),\n  PathConvNode(\'op2\')\n)'

        toy_path_list = PathList(Path(PathConvNode('op1')))
        assert repr(
            toy_path_list
        ) == 'PathList(\n  Path(\n    PathConvNode(\'op1\')\n  )\n)'

        path_list1 = PathList(Path(PathConvNode('op1')))
        path_list2 = PathList(Path(PathConvNode('op2')))
        toy_concat_node = PathConcatNode('op3', [path_list1, path_list2])
        assert repr(
            toy_concat_node
        ) == 'PathConcatNode(\n  PathList(\n    Path(\n      PathConvNode(\'op1\')\n    )\n  ),\n  PathList(\n    Path(\n      PathConvNode(\'op2\')\n    )\n  )\n)'  # noqa: E501

    def test_reset_bn_running_stats(self):
        _test_reset_bn_running_stats(False)
        with pytest.raises(AssertionError):
            _test_reset_bn_running_stats(True)

    def test_node(self):
        node1 = PathConvNode('conv1')
        node2 = PathConvNode('conv2')
        assert node1 != node2

        node1 = PathConvNode('conv1')
        node2 = PathConvNode('conv1')
        assert node1 == node2

    def test_path(self):
        node1 = PathConvNode('conv1')
        node2 = PathConvNode('conv2')

        path1 = Path([node1])
        path2 = Path([node2])
        assert path1 != path2

        path1 = Path([node1])
        path2 = Path([node1])
        assert path1 == path2

        assert path1[0] == node1

    def test_path_list(self):
        node1 = PathConvNode('conv1')
        node2 = PathConvNode('conv2')

        path1 = Path([node1])
        path2 = Path([node2])
        assert PathList(path1) == PathList([path1])
        assert PathList(path1) != PathList(path2)

        with self.assertRaisesRegex(AssertionError, ''):
            _ = PathList({})

    def test_sum_pseudo_loss(self):
        model = ResBlock()
        tracer = BackwardTracer(loss_calculator={'type': 'SumPseudoLoss'})
        path = tracer.trace(model)
        print(path)


def _test_reset_bn_running_stats(should_fail):
    import os
    import random

    import numpy as np

    def set_seed(seed: int) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(1024)
    imgs = torch.randn(2, 3, 4, 4)
    loss_calculator = ToyCNNPseudoLoss()
    tracer = BackwardTracer(loss_calculator=loss_calculator)
    if should_fail:
        tracer._reset_norm_running_stats = lambda *_: None

    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()

    model1 = ResBlock()
    set_seed(1)
    tracer.trace(model1)
    model1.eval()
    output1 = model1(imgs)

    set_seed(1024)
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    model2 = ResBlock()
    set_seed(2)
    tracer.trace(model2)
    model2.eval()
    output2 = model2(imgs)

    assert torch.equal(output1.norm(p='fro'), output2.norm(p='fro'))
