# Copyright (c) OpenMMLab. All rights reserved.
from copy import copy
from unittest import TestCase

import torch
from mmengine.model import BaseModule

try:
    from torch.ao.nn.intrinsic import ConvBnReLU2d
    from torch.ao.quantization.backend_config import BackendConfig
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quant_type import QuantType
except ImportError:
    from mmrazor.utils import get_placeholder
    ConvBnReLU2d = get_placeholder('torch>=1.13')
    BackendConfig = get_placeholder('torch>=1.13')
    PrepareCustomConfig = get_placeholder('torch>=1.13')
    ConObservedGraphModuleBnReLU2d = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    QuantType = get_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.quantizers import AcademicQuantizer
from mmrazor.models.quantizers.academic_quantizer import (
    FLOAT_TO_OBSERVED_DICT_KEY, GLOBAL_DICT_KEY, MODULE_NAME_DICT_KEY,
    OBJECT_TYPE_DICT_KEY, PRESERVED_ATTRIBUTES_DICT_KEY)
from mmrazor.registry import MODELS
from mmrazor.testing import ConvBNReLU


@MODELS.register_module()
class ToyFloatModel(BaseModule):

    def __init__(self) -> None:
        super().__init__()


@MODELS.register_module()
class ToyObservedModel(BaseModule):

    def __init__(self) -> None:
        super().__init__()


class TestAcademicQuantizer(TestCase):

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
        self.qconfig = dict(
            w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
            a_observer=dict(type='mmrazor.MinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.FakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
            w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=True),
            a_qscheme=dict(qdtype='quint8', bit=4, is_symmetry=True),
        )
        self.model = ConvBNReLU(3, 3, norm_cfg=dict(type='BN'))

    def test_gen_qconfig_mapping(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # test set GLOBAL_DICT_KEY by QConfigMapping
        global_qconfig = copy(self.global_qconfig)
        qconfig_mapping = {GLOBAL_DICT_KEY: global_qconfig}
        quantizer = AcademicQuantizer(qconfig_mapping=qconfig_mapping)
        assert hasattr(quantizer, 'qconfig_mapping')
        assert isinstance(quantizer.qconfig_mapping, QConfigMapping)
        assert quantizer.qconfig_mapping.global_qconfig

        # test set OBJECT_TYPE_DICT_KEY by QConfigMapping
        qconfig = copy(self.qconfig)
        qconfig_mapping = {
            OBJECT_TYPE_DICT_KEY:
            [('torch.ao.nn.intrinsic.ConvBnReLU2d', qconfig)]
        }
        quantizer = AcademicQuantizer(qconfig_mapping=qconfig_mapping)
        assert hasattr(quantizer, 'qconfig_mapping')
        assert isinstance(quantizer.qconfig_mapping, QConfigMapping)
        assert quantizer.qconfig_mapping.object_type_qconfigs.get(ConvBnReLU2d)

        # test set MODULE_NAME_DICT_KEY by QConfigMapping
        qconfig = copy(self.qconfig)
        qconfig_mapping = {
            MODULE_NAME_DICT_KEY: [('conv_module.conv', qconfig)]
        }
        quantizer = AcademicQuantizer(qconfig_mapping=qconfig_mapping)
        assert hasattr(quantizer, 'qconfig_mapping')
        assert isinstance(quantizer.qconfig_mapping, QConfigMapping)
        assert quantizer.qconfig_mapping.module_name_qconfigs.get(
            'conv_module.conv')

    def test_gen_prepare_custom_config(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        # test prepare_custom_config is None
        global_qconfig = copy(self.global_qconfig)
        qconfig_mapping = {GLOBAL_DICT_KEY: global_qconfig}
        quantizer = AcademicQuantizer(qconfig_mapping=qconfig_mapping)
        assert hasattr(quantizer, 'prepare_custom_config')
        assert isinstance(quantizer.prepare_custom_config, PrepareCustomConfig)

        # test set FLOAT_TO_OBSERVED_DICT_KEY and PRESERVED_ATTRIBUTES_DICT_KEY
        # by PrepareCustomConfig
        global_qconfig = copy(self.global_qconfig)
        qconfig_mapping = {GLOBAL_DICT_KEY: global_qconfig}
        flop_to_observed_list = [('ToyFloatModel', 'ToyObservedModel')]
        preserved_attributes_list = ['toy_attr1', 'toy_attr2']
        prepare_custom_config = {
            FLOAT_TO_OBSERVED_DICT_KEY: flop_to_observed_list,
            PRESERVED_ATTRIBUTES_DICT_KEY: preserved_attributes_list
        }
        quantizer = AcademicQuantizer(
            qconfig_mapping=qconfig_mapping,
            prepare_custom_config=prepare_custom_config)

        assert hasattr(quantizer, 'prepare_custom_config')
        assert isinstance(quantizer.prepare_custom_config, PrepareCustomConfig)
        mapping = quantizer.prepare_custom_config.float_to_observed_mapping[
            QuantType.STATIC]
        assert mapping.get(ToyFloatModel)
        assert mapping[ToyFloatModel] == ToyObservedModel

        attributes = quantizer.prepare_custom_config.preserved_attributes
        assert attributes == preserved_attributes_list

    def test_init(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        global_qconfig = copy(self.global_qconfig)
        qconfig_mapping = {GLOBAL_DICT_KEY: global_qconfig}
        quantizer = AcademicQuantizer(qconfig_mapping=qconfig_mapping)
        assert hasattr(quantizer, 'backend_config')
        assert isinstance(quantizer.backend_config, BackendConfig)

    def test_prepare(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')

        global_qconfig = copy(self.global_qconfig)
        qconfig_mapping = {GLOBAL_DICT_KEY: global_qconfig}
        preserved_attributes_list = ['toy_attr1', 'toy_attr2']
        prepare_custom_config = {
            PRESERVED_ATTRIBUTES_DICT_KEY: preserved_attributes_list
        }
        quantizer = AcademicQuantizer(
            qconfig_mapping=qconfig_mapping,
            prepare_custom_config=prepare_custom_config)
        model = copy(self.model)
        prepared = quantizer.prepare(model)
        assert isinstance(prepared, ObservedGraphModule)
        assert hasattr(prepared, 'toy_attr1')
        assert hasattr(prepared, 'toy_attr2')
