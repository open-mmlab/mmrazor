# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch

from mmrazor.models.task_modules.tracer import build_graphmodule
from mmrazor.models.utils import str2class
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import BackendConfigs, QConfigHandler
from .base import BaseQuantizer

try:
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.custom_config import (FuseCustomConfig,
                                                        PrepareCustomConfig)
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quantize_fx import _fuse_fx
except ImportError:
    from mmrazor.utils import get_placeholder
    prepare = get_placeholder('torch>=1.13')
    FuseCustomConfig = get_placeholder('torch>=1.13')
    PrepareCustomConfig = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')

GLOBAL_DICT_KEY = '_global_'
OBJECT_TYPE_DICT_KEY = 'object_type'
MODULE_NAME_DICT_KEY = 'module_name'

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
                ``module_name`` : sets the qconfig for modules matching the
                    given module name
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
                    `[('FloatCustomModule', 'ObservedCustomModule')]`
                ``preserved_attributes``: a list of attributes that persist
                    even if they are not used in ``forward``, e.g.
                    `['attr1', 'attr2']`
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

    def prepare(self, model, concrete_args=None):
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
        self.swap_ff_with_fxff(model)
        traced_graph = self.tracer.trace(model, concrete_args=concrete_args)
        graph_module = build_graphmodule(model, traced_graph)
        preserved_attributes = self.prepare_custom_config.preserved_attributes
        for attr_name in preserved_attributes:
            setattr(graph_module, attr_name, getattr(model, attr_name))
        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(
            preserved_attributes)

        # set the training modes of all modules to True to `_fuse_fx` correctly
        # todo: check freezebn
        self.sync_module_training_mode(graph_module, mode=True)

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
            conf.set_object_type(str2class(object_type), qconfig)

        for module_name, qconfig in qconfig_mapping.get(
                MODULE_NAME_DICT_KEY, []):
            qconfig = QConfigHandler(qconfig).convert()
            conf.set_module_name(module_name, qconfig)

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
            for float_class_str, observed_class_str in prepare_custom_config.get(  # noqa: E501
                    FLOAT_TO_OBSERVED_DICT_KEY, []):
                float_class = MODELS.get(float_class_str)
                observed_class = MODELS.get(observed_class_str)
                conf.set_float_to_observed_mapping(float_class, observed_class)
            conf.set_preserved_attributes(
                prepare_custom_config.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
            return conf
