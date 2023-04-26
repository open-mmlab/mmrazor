# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from mmengine.config import Config

try:
    from torch.ao.quantization import FakeQuantize, QConfig
except ImportError:
    from mmrazor.utils import get_placeholder
    QConfig = get_placeholder('torch>=1.13')
    FakeQuantize = get_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.fake_quants import register_torch_fake_quants
from mmrazor.models.observers import register_torch_observers
from mmrazor.registry import MODELS
from mmrazor.structures import QConfigHandler, QSchemeHandler

register_torch_observers()
register_torch_fake_quants()


class TestQSchemeHandler(TestCase):

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # per_channel
        qscheme = QSchemeHandler(is_symmetry=True, is_per_channel=True)
        assert qscheme.torch_qscheme is torch.per_channel_symmetric

        # per_tensor
        qscheme = QSchemeHandler(is_symmetry=True, is_per_channel=False)
        assert qscheme.torch_qscheme is torch.per_tensor_symmetric

        # qdtype is incorrect
        self.assertRaises(AssertionError, QSchemeHandler, 'float')

        # is_symmetric_range
        kwargs = {'is_symmetric_range': True}
        qscheme = QSchemeHandler(**kwargs)
        assert qscheme.is_symmetric_range is True

    def test_to_observer_params(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # qdtype = quint8
        ret_params = QSchemeHandler(qdtype='quint8').to_observer_params()
        assert ret_params['dtype'] == torch.quint8
        assert ret_params['quant_min'] == 0 and ret_params['quant_max'] == 255

        # qdtype = qint8, is_symmetric_range=False
        ret_params = QSchemeHandler(qdtype='qint8').to_observer_params()
        assert ret_params['dtype'] == torch.qint8
        assert ret_params['quant_min'] == -128 and ret_params[
            'quant_max'] == 127

        # qdtype = qint8, is_symmetric_range=True
        ret_params = QSchemeHandler(
            qdtype='qint8', is_symmetric_range=True).to_observer_params()
        assert ret_params['quant_min'] == -127 and ret_params[
            'quant_max'] == 127

        # per_channel
        ret_params = QSchemeHandler(is_per_channel=True).to_observer_params()
        assert ret_params['ch_axis'] == 0

        # per_tensor
        ret_params = QSchemeHandler(is_per_channel=False).to_observer_params()
        assert 'ch_axis' not in ret_params.keys()


class TestQConfigHandler(TestCase):

    def setUp(self):
        self.qconfig_dict = dict(
            w_observer=dict(type='MovingAveragePerChannelMinMaxObserver'),
            a_observer=dict(type='MovingAveragePerChannelMinMaxObserver'),
            w_fake_quant=dict(type='FakeQuantize'),
            a_fake_quant=dict(type='FakeQuantize'),
            w_qscheme=dict(
                qdtype='qint8',
                bit=8,
                is_symmetry=True,
                is_symmetric_range=True),
            a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
        )
        self.qconfig = Config(self.qconfig_dict)

    def test_check_qconfig(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        assert QConfigHandler.check_qconfig(self.qconfig_dict) is True
        assert QConfigHandler.check_qconfig(self.qconfig) is True
        qconfig_dict = copy.copy(self.qconfig_dict)
        print(qconfig_dict)
        qconfig_dict.pop('w_observer')
        assert QConfigHandler.check_qconfig(qconfig_dict) is False

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # test dict init
        qconfig = QConfigHandler(self.qconfig_dict)
        assert hasattr(qconfig, 'w_qscheme')
        assert hasattr(qconfig, 'a_qscheme')
        assert hasattr(qconfig, 'w_fake_quant')
        assert hasattr(qconfig, 'a_fake_quant')

        # test mmengine's Config init
        qconfig = QConfigHandler(self.qconfig)
        assert hasattr(qconfig, 'w_qscheme')
        assert hasattr(qconfig, 'a_qscheme')
        assert hasattr(qconfig, 'w_fake_quant')
        assert hasattr(qconfig, 'a_fake_quant')

        # per_channel
        assert qconfig.w_qscheme.is_per_channel is True
        assert qconfig.a_qscheme.is_per_channel is True

    def test_convert(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        qconfig = QConfigHandler(self.qconfig)
        torch_qconfig = qconfig.convert()
        assert isinstance(torch_qconfig, QConfig)

    def test_replace_fakequant(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # update_qparams is False
        qconfig = QConfigHandler(self.qconfig)
        org_fakequant_ins = qconfig.w_fake_quant()
        new_fakequant = qconfig.replace_fakequant(
            org_fakequant_ins, qconfig.w_qscheme, update_qparams=False)
        new_fakequant_ins = new_fakequant()
        assert isinstance(new_fakequant_ins, FakeQuantize)
        assert isinstance(new_fakequant_ins.activation_post_process,
                          MODELS.get('PerChannelMinMaxObserver'))

        # update_qparams is True
        qconfig = QConfigHandler(self.qconfig)
        org_fakequant_ins = qconfig.w_fake_quant()
        org_fakequant_ins.scale = torch.Tensor([2])
        org_fakequant_ins.activation_post_process.min_val = torch.Tensor([1])
        new_fakequant_ins = qconfig.replace_fakequant(
            org_fakequant_ins, qconfig.w_qscheme, update_qparams=True)
        assert isinstance(new_fakequant_ins, FakeQuantize)
        assert isinstance(new_fakequant_ins.activation_post_process,
                          MODELS.get('PerChannelMinMaxObserver'))
        assert new_fakequant_ins.scale == org_fakequant_ins.scale
        assert new_fakequant_ins.activation_post_process.min_val == \
            org_fakequant_ins.activation_post_process.min_val

    def test_fixed_w_fakequant(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        qconfig = QConfigHandler(self.qconfig)
        qconfig.fixed_w_fakequant()
        new_fakequant_ins = qconfig.w_fake_quant()
        assert isinstance(new_fakequant_ins, FakeQuantize)
        assert isinstance(new_fakequant_ins.activation_post_process,
                          MODELS.get('PerChannelMinMaxObserver'))
