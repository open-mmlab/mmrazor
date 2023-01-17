# Copyright (c) OpenMMLab. All rights reserved.
import operator
from unittest import TestCase

import torch
import torch.nn as nn

try:
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.quantize_fx import _fuse_fx
except ImportError:
    from mmrazor.utils import get_placeholder
    QConfigMapping = get_placeholder('torch>=1.13')
    FakeQuantizeBase = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.task_modules.tracer import CustomTracer, build_graphmodule
from mmrazor.models.task_modules.tracer.fx import (
    del_fakequant_after_function, del_fakequant_after_method,
    del_fakequant_after_module, del_fakequant_after_op,
    del_fakequant_before_function, del_fakequant_before_method,
    del_fakequant_before_module, del_fakequant_before_op)
from mmrazor.structures.quantization import BackendConfigs, QConfigHandler


def _get_attrs(target, attrs):
    attrs = attrs.split('.')

    for att in attrs:
        target = getattr(target, att, None)
    return target


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels

        self.norm1 = nn.BatchNorm2d(self.mid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, 1)
        self.conv2 = nn.Conv2d(self.mid_channels, out_channels, 1)

        self.relu = nn.ReLU6()
        self.drop_path = nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            out = self.drop_path(out)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem_layer = nn.Sequential(
            nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block = BasicBlock(3, 3)
        self.block2 = BasicBlock(3, 3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 4)

    def forward(self, x):
        x = self.stem_layer(x)
        x = self.maxpool(x)
        x = self.block(x)
        x = self.block2(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)


class TestGraphUtils(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        self.tracer = CustomTracer()
        self.backend_config = BackendConfigs['native']
        self.qconfig = QConfigHandler(global_qconfig)
        self.qconfig_mapping = QConfigMapping().set_global(
            self.qconfig.convert())
        self.example_inputs = (torch.randn(1, 3, 224, 224), )

    def swap_ff_with_fxff(self, model):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
                modules_to_swap.append(name)
            else:
                self.swap_ff_with_fxff(module)

        for name in modules_to_swap:
            del model._modules[name]
            model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()

    def test_del_fakequant_before_op(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        op_del_prev_fakequant = ('output', )

        prepared_after_del = del_fakequant_before_op(
            prepared, op_del_prev_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op in op_del_prev_fakequant:
                args = node.args
                self.assertIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op in op_del_prev_fakequant:
                args = node.args
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_before_op(
            prepared, op_del_prev_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op in op_del_prev_fakequant:
                args = node.args
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

    def test_del_fakequant_after_op(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        op_del_next_fakequant = ('placeholder', )

        prepared_after_del = del_fakequant_after_op(
            prepared, op_del_next_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op in op_del_next_fakequant:
                self.assertIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op in op_del_next_fakequant:
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_after_op(
            prepared, op_del_next_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op in op_del_next_fakequant:
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

    def test_del_fakequant_before_method(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        method_del_prev_fakequant = ('flatten', )

        prepared_after_del = del_fakequant_before_method(
            prepared, method_del_prev_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op == 'call_method' and \
                    node.target in method_del_prev_fakequant:
                args = node.args
                self.assertIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_method' and \
                    node.target in method_del_prev_fakequant:
                args = node.args
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_before_method(
            prepared, method_del_prev_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_method' and \
                    node.target in method_del_prev_fakequant:
                args = node.args
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

    def test_del_fakequant_after_method(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        method_del_next_fakequant = ('flatten', )

        prepared_after_del = del_fakequant_after_method(
            prepared, method_del_next_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op == 'call_method' and \
                    node.target in method_del_next_fakequant:
                self.assertIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_method' and \
                    node.target in method_del_next_fakequant:
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_after_method(
            prepared, method_del_next_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_method' and \
                    node.target in method_del_next_fakequant:
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

    def test_del_fakequant_before_function(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        function_del_prev_fakequant = (operator.add, )

        prepared_after_del = del_fakequant_before_function(
            prepared, function_del_prev_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op == 'call_function' and \
                    node.target in function_del_prev_fakequant:
                args = node.args
                self.assertIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_function' and \
                    node.target in function_del_prev_fakequant:
                args = node.args
                self.assertEqual(len(args), 2)
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[1].target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_before_function(
            prepared, function_del_prev_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_function' and \
                    node.target in function_del_prev_fakequant:
                args = node.args
                self.assertEqual(len(args), 2)
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)
                self.assertNotIsInstance(
                    _get_attrs(prepared, args[1].target), FakeQuantizeBase)

    def test_del_fakequant_after_function(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        function_del_next_fakequant = (operator.add, )

        prepared_after_del = del_fakequant_after_function(
            prepared, function_del_next_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op == 'call_function' and \
                    node.target in function_del_next_fakequant:
                self.assertIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_function' and \
                    node.target in function_del_next_fakequant:
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_after_function(
            prepared, function_del_next_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_function' and \
                    node.target in function_del_next_fakequant:
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

    def test_del_fakequant_before_module(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        module_del_prev_fakequant = (torch.nn.ReLU6, torch.nn.Identity)

        prepared_after_del = del_fakequant_before_module(
            prepared, module_del_prev_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op == 'call_module' and isinstance(
                    _get_attrs(prepared, node.target),
                    module_del_prev_fakequant):
                args = node.args
                self.assertIsInstance(
                    _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_module' and isinstance(
                    _get_attrs(prepared, node.target),
                    module_del_prev_fakequant):
                args = node.args
                if args[0].op == 'call_module':
                    self.assertNotIsInstance(
                        _get_attrs(prepared, args[0].target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_before_module(
            prepared, module_del_prev_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_module' and isinstance(
                    _get_attrs(prepared, node.target),
                    module_del_prev_fakequant):
                args = node.args
                if args[0].op == 'call_module':
                    self.assertNotIsInstance(
                        _get_attrs(prepared, args[0].target), FakeQuantizeBase)

    def test_del_fakequant_after_module(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        model_to_quantize = ToyModel()
        model_to_quantize.eval()

        self.swap_ff_with_fxff(model_to_quantize)
        traced_graph = self.tracer.trace(model_to_quantize)
        graph_module = build_graphmodule(model_to_quantize, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        module_del_next_fakequant = (torch.nn.MaxPool2d, )

        prepared_after_del = del_fakequant_after_module(
            prepared, module_del_next_fakequant, inplace=False)
        for node in prepared.graph.nodes:
            if node.op == 'call_module' and isinstance(
                    _get_attrs(prepared, node.target),
                    module_del_next_fakequant):
                self.assertIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_module' and isinstance(
                    _get_attrs(prepared, node.target),
                    module_del_next_fakequant):
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)

        prepared_after_del = del_fakequant_after_module(
            prepared, module_del_next_fakequant, inplace=True)
        for node in prepared_after_del.graph.nodes:
            if node.op == 'call_module' and isinstance(
                    _get_attrs(prepared, node.target),
                    module_del_next_fakequant):
                self.assertNotIsInstance(
                    _get_attrs(prepared, node.next.target), FakeQuantizeBase)
