# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from torch.fx import GraphModule

from mmrazor.models.algorithms import MMArchitectureQuant
from mmrazor.registry import MODELS


@MODELS.register_module()
class ToyQuantModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.act = nn.ReLU()

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            out = self.act(self.bn(self.conv(inputs)))
            return dict(loss=out)
        elif mode == 'predict':
            out = self.act(self.bn(self.conv(inputs))) + 1
            return out
        elif mode == 'tensor':
            out = self.act(self.bn(self.conv(inputs))) + 2
            return out


class TestMMArchitectureQuant(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        filename = 'fp_model.pth'
        filename = os.path.join(self.temp_dir, filename)
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
            float_checkpoint='fp_model.pth',
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
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        assert isinstance(self.toy_model, MMArchitectureQuant)
        assert hasattr(self.toy_model, 'quantizer')

    def test_sync_qparams(self):
        mode = self.toy_model.forward_modes[0]
        self.toy_model.sync_qparams(mode)
        w_loss = self.toy_model.qmodels['loss'].conv.state_dict()['weight']
        w_tensor = self.toy_model.qmodels['tensor'].conv.state_dict()['weight']
        w_pred = self.toy_model.qmodels['predict'].conv.state_dict()['weight']
        assert w_loss.equal(w_pred)
        assert w_loss.equal(w_tensor)

    def test_build_qmodels(self):
        for forward_modes in self.toy_model.forward_modes:
            qmodels = self.toy_model.qmodels[forward_modes]
            assert isinstance(qmodels, GraphModule)

    def test_calibrate_step(self):
        # TODO
        pass
