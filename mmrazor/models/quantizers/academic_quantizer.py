# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch

from mmrazor.registry import MODELS
from mmrazor.structures.quantization import BackendConfigs, QConfigHandler
from .base import BaseQuantizer

try:
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.custom_config import (FuseCustomConfig,
                                                        PrepareCustomConfig)
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quant_type import _quant_type_from_str
    from torch.ao.quantization.quantize_fx import _fuse_fx
except ImportError:
    from mmrazor.utils import get_placeholder
    prepare = get_placeholder('torch>=1.13')
    FuseCustomConfig = get_placeholder('torch>=1.13')
    PrepareCustomConfig = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    _quant_type_from_str = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')

GLOBAL_DICT_KEY = '_global_'
OBJECT_TYPE_DICT_KEY = 'object_type'
MODULE_NAME_REGEX_DICT_KEY = 'module_name_regex'
MODULE_NAME_DICT_KEY = 'module_name'
MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = 'module_name_object_type_order'

# keys can be used in `prepare_custom_config` of `AcademicQuantizer`.
FLOAT_TO_OBSERVED_DICT_KEY = 'float_to_observed_custom_module_class'
PRESERVED_ATTRIBUTES_DICT_KEY = 'preserved_attributes'


@MODELS.register_module()
class AcademicQuantizer(BaseQuantizer):
    """Quantizer for academic researching. Different from some quantizers for
    deploying, `AcademicQuantizer` is without the interfaces for deployment,
    but it has more flexible functions for quantizing your model. With its
    help, you can custom configuration qconfig for differenet OP by
    `qconfig_mapping` to implement customized experiments, including using
    custom fakquant, trying mixed precision quantization, comparing different
    quantization scheme and so on.

    Args:
        qconfig_mapping (Dict): Mapping from model ops to qconfig to configure
            how a model is quantized. You can specify qconfigs using the
            following keys (in increasing match priority):
                ``_global_`` : sets the global (default) qconfig
                ``object_type`` : sets the qconfig for a given module type,
                    function, or method name
                ``module_name_regex`` : sets the qconfig for modules matching
                    the given regex string
                ``module_name`` : sets the qconfig for modules matching the
                    given module name
                ``module_name_object_type_order`` : sets the qconfig for
                    modules matching a combination of the given module name,
                    object type, and the index at which the module appears
        tracer (Dict): It can be used to trace the float model to generate the
            corresponding graph, which contributes to prepare for quantizing
            the float model with code-free. Default to
            `dict(type='mmrazor.CustomTracer')`.
        prepare_custom_config (Optional[Dict]): Custom configuration for
            :func:`~torch.ao.quantization.fx.prepare`. You can specify the
            follow:
                ``float_to_observed_custom_module_class`` : a list of dict that
                    mapping from float module classes to observed module
                    classes, e.g.
                    `[dict(FloatCustomModule=ObservedCustomModule)]`
                ``preserved_attributes``: a list of attributes that persist
                    even if they are not used in ``forward``, e.g.
                    `[attr1, attr2]`
    """

    def __init__(self,
                 qconfig_mapping: Dict,
                 tracer: Dict = dict(type='mmrazor.CustomTracer'),
                 prepare_custom_config: Optional[Dict] = None):
        super().__init__(tracer)
        self.qconfig_mapping = self.gen_qconfig_mapping(qconfig_mapping)
        self.prepare_custom_config = self.gen_prepare_custom_config(
            prepare_custom_config)
        self.backend_config = BackendConfigs[self.backend]
        self.example_inputs = (torch.randn(1, 3, 224, 224), )

    @property
    def backend(self):
        """The key of the corresponding backend config."""
        return 'academic'

    def prepare(self, model, graph_module):
        """Prepare for quantizing model, which includes as follows:

        1. Swap floatfunctional with FXFloatFunctional;
        2. Trace model to generate `GraphModule`;
        2. Fuse some OPs combination, such as conv + bn, conv + relu and so on;
        3. Swap some conv or linear module with QAT Modules which contain
        weight fakequant nodes;
        4. Insert required fakequant nodes for activation.
        step 3 and step 4 are implemented in
        :func:`~torch.ao.quantization.fx.prepare`
        """
        preserved_attributes = self.prepare_custom_config.preserved_attributes
        for attr_name in preserved_attributes:
            setattr(graph_module, attr_name, getattr(model, attr_name))
        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(
            preserved_attributes)
        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            fuse_custom_config=fuse_custom_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            prepare_custom_config=self.prepare_custom_config,
            backend_config=self.backend_config)
        for attr_name in preserved_attributes:
            setattr(prepared, attr_name, getattr(model, attr_name))

        return prepared

    def gen_qconfig_mapping(self, qconfig_mapping: Dict):
        """Convert qconfig_mapping in config file to `QConfigMapping`.

        `QConfigMapping` is a custom class for mapping from model ops to
        :class:`torch.ao.quantization.QConfig` s.
        """
        conf = QConfigMapping()
        if GLOBAL_DICT_KEY in qconfig_mapping:
            qconfig = QConfigHandler(
                qconfig_mapping[GLOBAL_DICT_KEY]).convert()
            conf.set_global(qconfig)
        for object_type, qconfig in qconfig_mapping.get(
                OBJECT_TYPE_DICT_KEY, []):
            qconfig = QConfigHandler(qconfig).convert()
            conf.set_object_type(object_type, qconfig)

        for module_name_regex, qconfig in qconfig_mapping.get(
                MODULE_NAME_REGEX_DICT_KEY, []):
            qconfig = QConfigHandler(qconfig).convert()
            conf.set_module_name_regex(module_name_regex, qconfig)
        for module_name, qconfig in qconfig_mapping.get(
                MODULE_NAME_DICT_KEY, []):
            qconfig = QConfigHandler(qconfig).convert()
            conf.set_module_name(module_name, qconfig)
        for module_name, object_type, index, qconfig in qconfig_mapping.get(
                MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY, []):
            qconfig = QConfigHandler(qconfig).convert()
            conf.set_module_name_object_type_order(module_name, object_type,
                                                   index, qconfig)

        return conf

    def gen_prepare_custom_config(self, prepare_custom_config: Optional[Dict]):
        """Convert prepare_custom_config in config file to
        `PrepareCustomConfig`.

        `PrepareCustomConfig` is a custom class for custom configurating
        :func:`~torch.ao.quantization.fx.prepare`.
        """
        conf = PrepareCustomConfig()
        if prepare_custom_config is None:
            return conf
        else:
            for quant_type_name, custom_module_mapping in \
                prepare_custom_config.get(
                    FLOAT_TO_OBSERVED_DICT_KEY, {}).items():
                quant_type = _quant_type_from_str(quant_type_name)
                mapping_items = custom_module_mapping.items()
                for float_class_str, observed_class_str in mapping_items:
                    float_class = MODELS.get(float_class_str)
                    observed_class = MODELS.get(observed_class_str)
                    conf.set_float_to_observed_mapping(float_class,
                                                       observed_class,
                                                       quant_type)
            conf.set_preserved_attributes(
                prepare_custom_config.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
            return conf
