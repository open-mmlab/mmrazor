# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from mmengine.config import Config
from torch.ao.quantization import QConfig

from mmrazor.models.fake_quants import register_torch_fake_quants
from mmrazor.models.observers import register_torch_observers
from mmrazor.structures import QConfigHander, QSchemeHander

register_torch_observers()
register_torch_fake_quants()


class TestQSchemeHander(TestCase):

    def test_init(self):
        # per_channel
        qscheme = QSchemeHander(is_symmetry=True, is_per_channel=True)
        assert qscheme.torch_qscheme is torch.per_channel_symmetric

        # per_tensor
        qscheme = QSchemeHander(is_symmetry=True, is_per_channel=False)
        assert qscheme.torch_qscheme is torch.per_tensor_symmetric

        # qdtype is incorrect
        self.assertRaises(AssertionError, QSchemeHander, 'float')

        # is_symmetric_range
        kwargs = {'is_symmetric_range': True}
        qscheme = QSchemeHander(**kwargs)
        assert qscheme.is_symmetric_range is True

    def test_to_observer_params(self):
        # qdtype = quint8
        ret_params = QSchemeHander(qdtype='quint8').to_observer_params()
        assert ret_params['dtype'] == torch.quint8
        assert ret_params['quant_min'] == 0 and ret_params['quant_max'] == 255

        # qdtype = qint8, is_symmetric_range=False
        ret_params = QSchemeHander(qdtype='qint8').to_observer_params()
        assert ret_params['dtype'] == torch.qint8
        assert ret_params['quant_min'] == -128 and ret_params[
            'quant_max'] == 127

        # qdtype = qint8, is_symmetric_range=True
        ret_params = QSchemeHander(
            qdtype='qint8', is_symmetric_range=True).to_observer_params()
        assert ret_params['quant_min'] == -127 and ret_params[
            'quant_max'] == 127

        # per_channel
        ret_params = QSchemeHander(is_per_channel=True).to_observer_params()
        assert ret_params['ch_axis'] == 0

        # per_tensor
        ret_params = QSchemeHander(is_per_channel=False).to_observer_params()
        assert 'ch_axis' not in ret_params.keys()


class TestQConfigHander(TestCase):

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
        assert QConfigHander.check_qconfig(self.qconfig_dict) is True
        assert QConfigHander.check_qconfig(self.qconfig) is True
        qconfig_dict = copy.copy(self.qconfig_dict)
        print(qconfig_dict)
        qconfig_dict.pop('w_observer')
        assert QConfigHander.check_qconfig(qconfig_dict) is False

    def test_init(self):
        # test dict init
        qconfig = QConfigHander(self.qconfig_dict)
        assert hasattr(qconfig, 'w_qscheme')
        assert hasattr(qconfig, 'a_qscheme')
        assert hasattr(qconfig, 'w_fake_quant')
        assert hasattr(qconfig, 'a_fake_quant')

        # test mmengine's Config init
        qconfig = QConfigHander(self.qconfig)
        assert hasattr(qconfig, 'w_qscheme')
        assert hasattr(qconfig, 'a_qscheme')
        assert hasattr(qconfig, 'w_fake_quant')
        assert hasattr(qconfig, 'a_fake_quant')

        # per_channel
        assert qconfig.w_qscheme.is_per_channel is True
        assert qconfig.a_qscheme.is_per_channel is True

    def test_convert(self):
        qconfig = QConfigHander(self.qconfig)
        torch_qconfig = qconfig.convert()
        assert isinstance(torch_qconfig, QConfig)
