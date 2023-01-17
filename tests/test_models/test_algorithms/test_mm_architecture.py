# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn

try:
    from torch.fx import GraphModule
except ImportError:
    from mmrazor.utils import get_placeholder
    GraphModule = get_placeholder('torch>=1.13')

from mmengine.model import BaseModel

from mmrazor import digit_version
from mmrazor.models.algorithms import MMArchitectureQuant
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


@MODELS.register_module()
class ToyQuantModel(BaseModel):

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


class TestMMArchitectureQuant(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.temp_dir = tempfile.mkdtemp()
        filename = 'fp_model.pth'
        filename = os.path.join(self.temp_dir, filename)
        # import pdb; pdb.set_trace()
        toymodel = ToyQuantModel()
        torch.save(toymodel.state_dict(), filename)

        global_qconfig = dict(
            w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
            a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.FakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
            w_qscheme=dict(
                qdtype='qint8',
                bit=8,
                is_symmetry=True,
                is_symmetric_range=True),
            a_qscheme=dict(
                qdtype='quint8',
                bit=8,
                is_symmetry=True,
                averaging_constant=0.1),
        )
        alg_kwargs = dict(
            type='mmrazor.MMArchitectureQuant',
            architecture=dict(type='ToyQuantModel'),
            float_checkpoint=filename,
            quantizer=dict(
                type='mmrazor.OpenVINOQuantizer',
                global_qconfig=global_qconfig,
                tracer=dict(
                    type='mmrazor.CustomTracer',
                    skipped_methods=[
                        'mmcls.models.heads.ClsHead._get_loss',
                        'mmcls.models.heads.ClsHead._get_predictions'
                    ])))
        self.alg_kwargs = alg_kwargs
        self.toy_model = MODELS.build(self.alg_kwargs)

    def tearDown(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        assert isinstance(self.toy_model, MMArchitectureQuant)
        assert hasattr(self.toy_model, 'quantizer')

    def test_sync_qparams(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        mode = self.toy_model.forward_modes[0]
        self.toy_model.sync_qparams(mode)
        w_loss = self.toy_model.qmodels['loss'].block.conv1.state_dict(
        )['weight']
        w_tensor = self.toy_model.qmodels['tensor'].block.conv1.state_dict(
        )['weight']
        w_pred = self.toy_model.qmodels['predict'].block.conv1.state_dict(
        )['weight']
        assert w_loss.equal(w_pred)
        assert w_loss.equal(w_tensor)

    def test_build_qmodels(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        for forward_modes in self.toy_model.forward_modes:
            qmodels = self.toy_model.qmodels[forward_modes]
            assert isinstance(qmodels, GraphModule)

    def test_calibrate_step(self):
        # TODO
        pass
