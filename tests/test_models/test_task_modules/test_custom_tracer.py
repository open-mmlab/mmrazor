# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from mmcls.models.backbones.resnet import ResLayer
from mmengine.config import Config
from mmengine.registry import MODELS

try:
    from torch.fx import GraphModule
    from torch.fx._symbolic_trace import Graph
except ImportError:
    from mmrazor.utils import get_placeholder
    GraphModule = get_placeholder('torch>=1.13')
    Graph = get_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.task_modules.tracer import (CustomTracer,
                                                UntracedMethodRegistry,
                                                build_graphmodule,
                                                custom_symbolic_trace)
from mmrazor.models.task_modules.tracer.fx.custom_tracer import \
    _prepare_module_dict


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def get_loss(self, x):
        return x * 0.1

    def extrac_feature(self, x):
        return x * 2

    def forward(self, x):
        x = self.extrac_feature(x)
        x = self.get_loss(x)
        return x


class testUntracedMethodRgistry(TestCase):

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        method = ToyModel.get_loss
        method_registry = UntracedMethodRegistry(method)
        assert hasattr(method_registry, 'method')
        assert hasattr(method_registry, 'method_dict')

    def test_registry_method(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model = ToyModel
        method = ToyModel.get_loss
        method_registry = UntracedMethodRegistry(method)
        method_registry.__set_name__(model, 'get_loss')
        assert 'get_loss' in method_registry.method_dict.keys()
        assert method_registry.method_dict['get_loss']['mod'] == model


class testCustomTracer(TestCase):

    def setUp(self):
        self.cfg = Config.fromfile(
            'tests/data/test_models/test_task_modules/mmcls_cfg.py')
        self.skipped_methods = [
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ]
        self.skipped_module_names = ['backbone.layer4.0']
        self.skipped_module_classes = [ResLayer]

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # init without skipped_methods
        tracer = CustomTracer()
        assert hasattr(tracer, 'skipped_methods')
        assert len(tracer.skipped_methods) == 0
        # init with skipped_methods(list)
        UntracedMethodRegistry.method_dict = dict()
        tracer = CustomTracer(skipped_methods=self.skipped_methods)
        assert '_get_loss' in UntracedMethodRegistry.method_dict.keys()
        assert '_get_predictions' in UntracedMethodRegistry.method_dict.keys()
        # init with skipped_methods(str)
        UntracedMethodRegistry.method_dict = dict()
        tracer = CustomTracer(skipped_methods=self.skipped_methods[0])
        assert '_get_loss' in UntracedMethodRegistry.method_dict.keys()
        # init with skipped_methods(int, error)
        with self.assertRaises(TypeError):
            CustomTracer(skipped_methods=123)
        # init with skipped_methods(str, error)
        with self.assertRaises(AssertionError):
            CustomTracer(skipped_methods='_get_loss')

    def test_trace(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # test trace with skipped_methods
        model = MODELS.build(self.cfg.model)
        UntracedMethodRegistry.method_dict = dict()
        tracer = CustomTracer(skipped_methods=self.skipped_methods)
        graph_tensor = tracer.trace(model, concrete_args={'mode': 'tensor'})
        graph_loss = tracer.trace(model, concrete_args={'mode': 'loss'})
        graph_predict = tracer.trace(model, concrete_args={'mode': 'predict'})
        assert isinstance(graph_tensor, Graph)
        assert isinstance(graph_loss, Graph)
        skip_flag_loss = False
        for node in graph_loss.nodes:
            if node.op == 'call_method' and node.target == '_get_loss':
                skip_flag_loss = True
        assert isinstance(graph_predict, Graph)
        skip_flag_predict = False
        for node in graph_predict.nodes:
            if node.op == 'call_method' and node.target == '_get_predictions':
                skip_flag_predict = True
        assert skip_flag_loss and skip_flag_predict

        # test trace with skipped_module_names
        model = MODELS.build(self.cfg.model)
        UntracedMethodRegistry.method_dict = dict()
        tracer = CustomTracer(skipped_module_names=self.skipped_module_names)
        graph_tensor = tracer.trace(model, concrete_args={'mode': 'tensor'})
        skip_flag = False
        for node in graph_tensor.nodes:
            skipped_module_name = self.skipped_module_names[0]
            if node.op == 'call_module' and node.target == skipped_module_name:
                skip_flag = True
        assert skip_flag

        # test trace with skipped_module_classes
        model = MODELS.build(self.cfg.model)
        UntracedMethodRegistry.method_dict = dict()
        tracer = CustomTracer(
            skipped_module_classes=self.skipped_module_classes)
        graph_tensor = tracer.trace(model, concrete_args={'mode': 'tensor'})
        skip_flag = False
        for node in graph_tensor.nodes:
            if node.op == 'call_module' and node.target == 'backbone.layer1':
                skip_flag = True
        assert skip_flag


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    reason='version of torch < 1.13.0')
def test_custom_symbolic_trace():
    cfg = Config.fromfile(
        'tests/data/test_models/test_task_modules/mmcls_cfg.py')
    model = MODELS.build(cfg.model)
    UntracedMethodRegistry.method_dict = dict()
    graph_module = custom_symbolic_trace(
        model, concrete_args={'mode': 'tensor'})
    assert isinstance(graph_module, GraphModule)


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    reason='version of torch < 1.13.0')
def test_build_graphmodule():
    skipped_methods = ['mmcls.models.heads.ClsHead._get_predictions']
    cfg = Config.fromfile(
        'tests/data/test_models/test_task_modules/mmcls_cfg.py')
    model = MODELS.build(cfg.model)
    UntracedMethodRegistry.method_dict = dict()
    tracer = CustomTracer(skipped_methods=skipped_methods)
    graph_predict = tracer.trace(model, concrete_args={'mode': 'predict'})
    graph_module = build_graphmodule(model, graph_predict)
    assert isinstance(graph_module, GraphModule)

    # test _prepare_module_dict
    modules = dict(model.named_modules())
    module_dict = _prepare_module_dict(model, graph_predict)
    for k, v in module_dict.items():
        assert isinstance(v, torch.nn.Module)
        assert not isinstance(v, modules[k].__class__)
