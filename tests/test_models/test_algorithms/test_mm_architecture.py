# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch.nn as nn

from mmengine import ConfigDict
from mmengine.model import BaseModel

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
    def test_init(self):
        # _base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']
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
        alg_kwargs = ConfigDict(
            type = 'mmrazor.MMArchitectureQuant',
            architecture=dict(type='ToyQuantModel'),
            # architecture=dict(type='mmcls.ResNet', depth=18),
            quantizer=dict(
                type='mmrazor.OpenVINOQuantizer',
                global_qconfig=global_qconfig,
                tracer=dict(
                    type='mmrazor.CustomTracer',
                    skipped_methods=[
                        'mmcls.models.heads.ClsHead._get_loss',
                        'mmcls.models.heads.ClsHead._get_predictions'
                    ])))
        toy_model=MODELS.build(alg_kwargs)
        assert isinstance(toy_model, MMArchitectureQuant)
        assert hasattr(toy_model, 'quantizer')
        # _ = MMArchitectureQuant()
            
    def test_sync_qparams(self):
        # self.test_sync_qparams()
        pass
    
    def test_build_qmodels(self):
        
        pass
    
    def test_calibrate_step(self):
        pass