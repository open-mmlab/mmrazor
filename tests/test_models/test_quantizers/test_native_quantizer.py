# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor import digit_version
from mmrazor.models.quantizers import TorchNativeQuantizer
from mmrazor.models.quantizers.native_quantizer import SUPPORT_QAT_MODULES
from mmrazor.models.task_modules.tracer import CustomTracer
from mmrazor.models.task_modules.tracer.fx.custom_tracer import \
    build_graphmodule
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import BackendConfigs, QConfigHandler

try:
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quantize_fx import _fuse_fx
    from torch.fx import GraphModule
except ImportError:
    from mmrazor.utils import get_placeholder
    GraphModule = get_placeholder('torch>=1.13')
    ObservedGraphModule = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')


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


class ToyQuantModel(nn.Module):

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
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1))

no_observer_modules = [
    'torch.nn.Conv2d',
]

q_kwargs = dict(
    type='mmrazor.TorchNativeQuantizer',
    global_qconfig=global_qconfig,
    no_observer_modules=no_observer_modules,
    tracer=dict(type='CustomTracer'),
)


class TestTorchNativeQuantizer(TestCase):
    """TODO.

    Args:
        TestCase (_type_): _description_
    """

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.q_kwargs = q_kwargs
        self.tracer = CustomTracer()
        self.backend_config = BackendConfigs['native']
        self.qconfig = QConfigHandler(global_qconfig)
        self.qconfig_mapping = QConfigMapping().set_global(
            self.qconfig.convert())
        self.example_inputs = (torch.randn(1, 3, 224, 224), )
        self.native_quantizer = MODELS.build(self.q_kwargs)

    def tearDown(self):
        pass

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

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        native_quantizer = MODELS.build(self.q_kwargs)
        self.assertIsInstance(native_quantizer, TorchNativeQuantizer)

    def test_prepare(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        toy_model = ToyQuantModel()
        toy_model.eval()

        self.swap_ff_with_fxff(toy_model)
        traced_graph = self.tracer.trace(toy_model)
        graph_module = build_graphmodule(toy_model, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        assert isinstance(graph_module, GraphModule)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)
        assert isinstance(prepared, ObservedGraphModule)

        prepared = self.native_quantizer.del_redundant_fakequant(prepared)
        assert isinstance(prepared, GraphModule)

    def post_process_for_deploy(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        toy_model = ToyQuantModel()
        toy_model.eval()

        self.swap_ff_with_fxff(toy_model)
        traced_graph = self.tracer.trace(toy_model)
        graph_module = build_graphmodule(toy_model, traced_graph)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        assert isinstance(graph_module, GraphModule)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)
        assert isinstance(prepared, ObservedGraphModule)

        prepared = self.native_quantizer.del_redundant_fakequant(prepared)
        assert isinstance(prepared, GraphModule)

        prepared_no_fq = prepared

        self.native_quantizer.post_process_weight_fakequant(prepared)
        for name, child in prepared.named_children():
            if isinstance(child, SUPPORT_QAT_MODULES):
                raise ValueError
        self.native_quantizer.post_process_weight_fakequant(
            prepared_no_fq, True)
        for name, child in prepared_no_fq.named_children():
            if isinstance(child, SUPPORT_QAT_MODULES):
                raise ValueError
