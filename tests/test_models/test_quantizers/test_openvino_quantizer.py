# Copyright (c) OpenMMLab. All rights reserved.
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
from mmrazor.models.quantizers import OpenVINOQuantizer
from mmrazor.testing import ConvBNReLU


class TestOpenVINOQuantizer(TestCase):

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
        quantizer = OpenVINOQuantizer(global_qconfig=global_qconfig)
        assert quantizer.backend == 'openvino'
        assert quantizer.support_w_modes == ('per_tensor', 'per_channel')
        assert quantizer.support_a_modes == ('per_tensor')
        assert quantizer.module_prev_wo_fakequant
        assert quantizer.module_next_wo_fakequant
        assert quantizer.method_next_wo_fakequant
        assert quantizer.op_prev_wo_fakequant
