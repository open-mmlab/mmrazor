# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import shutil
import tempfile
from unittest import TestCase, skipIf

import torch
import torch.nn as nn

try:
    import onnx
    from onnx import helper
    from torch.fx import GraphModule
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    GraphModule = get_placeholder('torch>=1.13')
    onnx = get_package_placeholder('No module named onnx')
    helper = get_package_placeholder('No module named onnx.helper')

from mmengine import ConfigDict
from mmengine.model import BaseModel

try:
    import mmdeploy
except ImportError:
    from mmrazor.utils import get_package_placeholder
    mmdeploy = get_package_placeholder('mmdeploy')

from mmrazor import digit_version
from mmrazor.models.quantizers.exporters import (OpenVinoQuantizeExportor,
                                                 TensorRTExplicitExporter)
from mmrazor.models.quantizers.exporters.optim_utils import ONNXOptimUtils
from mmrazor.registry import MODELS


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
        super(ToyModel, self).__init__()
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


class ToyQuantModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.architecture = ToyModel()

    def loss(self, outputs, data_samples):
        return dict(loss=outputs.sum() - data_samples.sum())

    def forward(self, inputs, data_samples, mode: str = 'tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        outputs = self.architecture(inputs)

        return outputs


OpenVINO_GLOBAL_QCONFIG = ConfigDict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

OpenVINO_ALG_CONFIG = ConfigDict(
    type='mmrazor.MMArchitectureQuant',
    architecture=dict(type='ToyQuantModel'),
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=OpenVINO_GLOBAL_QCONFIG,
        tracer=dict(type='mmrazor.CustomTracer')))

TensorRT_GLOBAL_QCONFIG = ConfigDict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

TensorRT_ALG_CONFIG = ConfigDict(
    type='mmrazor.MMArchitectureQuant',
    architecture=dict(type='ToyQuantModel'),
    quantizer=dict(
        type='mmrazor.TensorRTQuantizer',
        global_qconfig=OpenVINO_GLOBAL_QCONFIG,
        tracer=dict(type='mmrazor.CustomTracer')))


@skipIf(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    'PyTorch version lower than 1.13.0 is not supported.')
class TestONNXOptimUtils(TestCase):

    def setUp(self):
        MODELS.register_module(module=ToyQuantModel, force=True)
        self.temp_dir = tempfile.mkdtemp()
        filename = 'symbolic.onnx'
        filename = os.path.join(self.temp_dir, filename)
        toy_model = MODELS.build(OpenVINO_ALG_CONFIG)
        observed_model = toy_model.get_deploy_model()
        torch.onnx.export(
            observed_model,
            torch.rand(2, 3, 16, 16),
            filename,
            opset_version=11)
        self.onnx_model = onnx.load(filename)
        self.optimizer = ONNXOptimUtils

    def tearDown(self):
        MODELS.module_dict.pop('ToyQuantModel')
        shutil.rmtree(self.temp_dir)

    def test_map_name_and_data(self):
        params = self.optimizer.map_name_and_data(self.onnx_model)
        params_keys = [
            'activation_post_process_0.scale',
            'activation_post_process_0.zero_point',
            'architecture.stem_layer.0.weight',
            'architecture.stem_layer.0.bias',
            'architecture.stem_layer.0.weight_fake_quant.scale',
            'architecture.stem_layer.0.weight_fake_quant.zero_point',
            'architecture.block.conv1.weight', 'architecture.block.conv1.bias',
            'architecture.block.conv1.weight_fake_quant.scale',
            'architecture.block.conv2.bias',
            'architecture.block2.conv1.weight',
            'architecture.block2.conv1.bias',
            'architecture.block2.conv1.weight_fake_quant.scale',
            'architecture.block2.conv2.weight',
            'architecture.block2.conv2.bias',
            'architecture.block2.conv2.weight_fake_quant.scale',
            'architecture.fc.weight', 'architecture.fc.bias',
            'architecture.fc.weight_fake_quant.scale',
            'architecture.fc.weight_fake_quant.zero_point',
            'activation_post_process_15.zero_point',
            'activation_post_process_15.scale',
            'activation_post_process_14.zero_point',
            'activation_post_process_14.scale',
            'activation_post_process_12.zero_point',
            'activation_post_process_12.scale',
            'activation_post_process_10.zero_point',
            'activation_post_process_10.scale',
            'activation_post_process_8.zero_point',
            'activation_post_process_8.scale',
            'activation_post_process_6.zero_point',
            'activation_post_process_6.scale',
            'activation_post_process_4.zero_point',
            'activation_post_process_4.scale',
            'activation_post_process_1.zero_point',
            'activation_post_process_1.scale',
            'architecture.block2.conv2.weight_fake_quant.zero_point',
            'architecture.block2.conv1.weight_fake_quant.zero_point',
            'architecture.block.conv2.weight_fake_quant.zero_point',
            'architecture.block.conv2.weight_fake_quant.scale',
            'architecture.block.conv2.weight',
            'architecture.block.conv1.weight_fake_quant.zero_point',
            '/activation_post_process_0/Constant_output_0',
            '/activation_post_process_0/Constant_1_output_0',
            '/stem_layer.0/weight_fake_quant/Constant_output_0',
            '/stem_layer.0/weight_fake_quant/Constant_1_output_0',
            '/relu/Constant_output_0', '/relu/Constant_1_output_0',
            '/relu_dup1/Constant_output_0', '/relu_dup1/Constant_1_output_0',
            '/relu_1/Constant_output_0', '/relu_1/Constant_1_output_0',
            '/relu_dup1_1/Constant_output_0',
            '/relu_dup1_1/Constant_1_output_0'
        ]
        self.assertEqual(set(params.keys()), set(params_keys))

    def test_map_name_and_initializer(self):
        initializers = self.optimizer.map_name_and_initializer(self.onnx_model)
        for init in self.onnx_model.graph.initializer:
            self.assertIn(init.name, initializers.keys())
        # self.assertEqual(set(initializers.keys()), set(initializers_keys))

    def test_map_output_and_node(self):
        _ = self.optimizer.map_output_and_node(self.onnx_model)

    def test_map_input_and_node(self):
        _ = self.optimizer.map_input_and_node(self.onnx_model)

    def test_remove_node_from_onnx(self):
        onnx_model = copy.deepcopy(self.onnx_model)
        node_to_remove = next(iter(onnx_model.graph.node))
        self.optimizer.remove_node_from_onnx(node_to_remove, onnx_model)
        for node in onnx_model.graph.node:
            self.assertNotEqual(node, node_to_remove)

    def test_remove_initializer_from_onnx(self):
        onnx_model = copy.deepcopy(self.onnx_model)
        initializer_to_remove = next(iter(onnx_model.graph.initializer))
        self.optimizer.remove_initializer_from_onnx(initializer_to_remove,
                                                    onnx_model)
        for initializer in onnx_model.graph.initializer:
            self.assertNotEqual(initializer, initializer_to_remove)

    def test_find_standalone_nodes(self):
        standalone_nodes = self.optimizer.find_standalone_nodes(
            self.onnx_model)
        self.assertEqual(standalone_nodes, [])

    def test_find_redundant_initializers(self):
        redundant_initializers = self.optimizer.find_redundant_initializers(
            self.onnx_model)
        self.assertEqual(redundant_initializers, [])

    def test_topo_sort(self):
        onnx_model = copy.deepcopy(self.onnx_model)
        onnx_model_topo_sort = self.optimizer.topo_sort(onnx_model)
        self.assertEqual(
            len(onnx_model_topo_sort.graph.node),
            len(self.onnx_model.graph.node))

    def test_optimize(self):
        onnx_model = copy.deepcopy(self.onnx_model)
        fake_node = helper.make_node('fake_node', [], [], mode='constant')
        self.optimizer.insert_node_to_onnx(fake_node, onnx_model)
        self.optimizer.optimize(onnx_model)
        for node in onnx_model.graph.node:
            self.assertNotEqual(node, fake_node)


@skipIf(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    'PyTorch version lower than 1.13.0 is not supported.')
class TestOpenVinoQuantizeExportor(TestCase):

    def setUp(self):
        MODELS.register_module(module=ToyQuantModel, force=True)
        self.temp_dir = tempfile.mkdtemp()
        filename = 'toy_model_symbolic.onnx'
        filename = os.path.join(self.temp_dir, filename)
        toy_model = MODELS.build(OpenVINO_ALG_CONFIG)
        observed_model = toy_model.get_deploy_model()
        torch.onnx.export(
            observed_model,
            torch.rand(2, 3, 16, 16),
            filename,
            opset_version=11)
        self.onnx_model = onnx.load(filename)
        self.export_path = os.path.join(self.temp_dir, 'toy_model.onnx')

    def tearDown(self):
        MODELS.module_dict.pop('ToyQuantModel')
        shutil.rmtree(self.temp_dir)

    def test_export(self):
        exporter = OpenVinoQuantizeExportor(self.onnx_model, self.export_path)
        exporter.export()
        self.assertTrue(os.path.exists(self.export_path))
        onnx_model = onnx.load(self.export_path)
        self.assertIsInstance(onnx_model, onnx.ModelProto)


@skipIf(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    'PyTorch version lower than 1.13.0 is not supported.')
class TestTensorRTExplicitExporter(TestCase):

    def setUp(self):
        MODELS.register_module(module=ToyQuantModel, force=True)
        self.temp_dir = tempfile.mkdtemp()
        filename = 'toy_model_symbolic.onnx'
        filename = os.path.join(self.temp_dir, filename)
        toy_model = MODELS.build(TensorRT_ALG_CONFIG)
        observed_model = toy_model.get_deploy_model()
        torch.onnx.export(
            observed_model,
            torch.rand(2, 3, 16, 16),
            filename,
            opset_version=11)
        self.onnx_model = onnx.load(filename)
        self.export_path = os.path.join(self.temp_dir, 'toy_model.onnx')

    def tearDown(self):
        MODELS.module_dict.pop('ToyQuantModel')
        shutil.rmtree(self.temp_dir)

    def test_export(self):
        exporter = TensorRTExplicitExporter(self.onnx_model, self.export_path)
        exporter.export()
        self.assertTrue(os.path.exists(self.export_path))
        onnx_model = onnx.load(self.export_path)
        self.assertIsInstance(onnx_model, onnx.ModelProto)
