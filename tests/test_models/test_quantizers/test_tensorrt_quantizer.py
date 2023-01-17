# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import tempfile
from copy import copy
from unittest import TestCase

import torch

try:
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
except ImportError:
    from mmrazor.utils import get_placeholder
    ObservedGraphModule = get_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.quantizers import TensorRTQuantizer
from mmrazor.testing import ConvBNReLU


class TestTensorRTQuantizer(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        self.global_qconfig = dict(
            w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
            a_observer=dict(type='mmrazor.MinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.FakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
            w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
            a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
        )
        self.temp_dir = tempfile.mkdtemp()
        self.model = ConvBNReLU(3, 3, norm_cfg=dict(type='BN'))

    def tearDown(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        shutil.rmtree(self.temp_dir)

    def test_property(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        global_qconfig = copy(self.global_qconfig)
        quantizer = TensorRTQuantizer(global_qconfig=global_qconfig)
        assert quantizer.backend == 'tensorrt'
        assert quantizer.support_w_modes == ('per_tensor', 'per_channel')
        assert quantizer.support_a_modes == ('per_tensor')

    def test_prepare_for_mmdeploy(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        global_qconfig = copy(self.global_qconfig)
        quantizer = TensorRTQuantizer(global_qconfig=global_qconfig)
        model = copy(self.model)

        # test checkpoint is None
        prepared_deploy = quantizer.prepare_for_mmdeploy(model=model)
        assert isinstance(prepared_deploy, ObservedGraphModule)

        # test checkpoint is not None
        ckpt_path = os.path.join(self.temp_dir,
                                 'test_prepare_for_mmdeploy.pth')
        model = copy(self.model)
        prepared = quantizer.prepare(model)
        torch.save({'state_dict': prepared.state_dict()}, ckpt_path)
        prepared_deploy = quantizer.prepare_for_mmdeploy(
            model=model, checkpoint=ckpt_path)
        assert isinstance(prepared_deploy, ObservedGraphModule)
