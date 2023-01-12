# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch

try:
    from torch.ao.quantization import enable_fake_quant
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.qconfig_mapping import (
        _FIXED_QPARAMS_OP_TO_OBSERVER, FixedQParamsFakeQuantize, QConfig,
        QConfigMapping, default_weight_fake_quant)
    from torch.ao.quantization.quantize_fx import _fuse_fx
    from torch.nn.intrinsic.qat import modules as qat_fused_modules
    from torch.nn.qat import modules as qat_modules
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    enable_fake_quant = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')
    qat_fused_modules = get_package_placeholder('torch>=1.13')
    qat_modules = get_package_placeholder('torch>=1.13')
    _FIXED_QPARAMS_OP_TO_OBSERVER = get_package_placeholder('torch>=1.13')
    FixedQParamsFakeQuantize = get_package_placeholder('torch>=1.13')
    QConfig = get_package_placeholder('torch>=1.13')
    default_weight_fake_quant = get_package_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.task_modules.tracer.fx import (
    del_fakequant_after_function, del_fakequant_after_method,
    del_fakequant_after_module, del_fakequant_after_op,
    del_fakequant_before_function, del_fakequant_before_method,
    del_fakequant_before_module, del_fakequant_before_op)
from mmrazor.models.utils import str2class
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import BackendConfigs, QConfigHander
from .base import BaseQuantizer

if digit_version(torch.__version__) >= digit_version('1.13.0'):
    SUPPORT_QAT_MODULES: Tuple = (
        qat_fused_modules.ConvBn1d, qat_fused_modules.ConvBn2d,
        qat_fused_modules.ConvBn3d, qat_fused_modules.ConvBnReLU1d,
        qat_fused_modules.ConvBnReLU2d, qat_fused_modules.ConvBnReLU3d,
        qat_fused_modules.ConvReLU1d, qat_fused_modules.ConvReLU2d,
        qat_fused_modules.ConvReLU3d, qat_fused_modules.LinearBn1d,
        qat_fused_modules.LinearReLU, qat_modules.Conv1d, qat_modules.Conv2d,
        qat_modules.Conv3d, qat_modules.Linear)

    MERGE_BN_MAPPINGS: Dict = {
        qat_fused_modules.ConvBn1d: qat_modules.Conv1d,
        qat_fused_modules.ConvBn2d: qat_modules.Conv2d,
        qat_fused_modules.ConvBn3d: qat_modules.Conv3d,
        qat_fused_modules.ConvBnReLU1d: qat_fused_modules.ConvReLU1d,
        qat_fused_modules.ConvBnReLU2d: qat_fused_modules.ConvReLU2d,
        qat_fused_modules.ConvBnReLU3d: qat_fused_modules.ConvReLU3d,
        qat_fused_modules.LinearBn1d: qat_modules.Linear
    }
else:
    SUPPORT_QAT_MODULES = ()
    MERGE_BN_MAPPINGS = {}


@MODELS.register_module()
class NativeQuantizer(BaseQuantizer):
    """tmp."""

    # backend: 'native'
    # support_w_modes = ['per_tensor', 'per_channel']
    # support_a_modes = ['per_tensor']

    def __init__(self,
                 is_qat,
                 global_qconfig,
                 no_observer_modules=None,
                 tracer=dict(type='CustomTracer'),
                 extra_redundant_fakequants=dict(
                     extra_module_prev_wo_fakequant=tuple(),
                     extra_module_next_wo_fakequant=tuple(),
                     extra_function_prev_wo_fakequant=tuple(),
                     extra_function_next_wo_fakequant=tuple(),
                     extra_method_prev_wo_fakequant=tuple(),
                     extra_method_next_wo_fakequant=tuple(),
                     extra_op_prev_wo_fakequant=tuple(),
                     extra_op_next_wo_fakequant=tuple())):
        super().__init__(tracer, is_qat)
        self.qconfig = QConfigHander(global_qconfig)
        if self.qconfig.w_qscheme.is_per_channel:
            w_mode = 'per_channel'
        else:
            w_mode = 'per_tensor'
        if self.qconfig.a_qscheme.is_per_channel:
            a_mode = 'per_channel'
        else:
            a_mode = 'per_tensor'
        assert w_mode in self.support_w_modes
        assert a_mode in self.support_a_modes

        self.qconfig_mapping = QConfigMapping().set_global(
            self.qconfig.convert())
        if no_observer_modules:
            self.no_observer_modules = str2class(no_observer_modules)
            for mod in self.no_observer_modules:
                self.qconfig_mapping.set_object_type(mod, None)
        else:
            self.no_observer_modules = no_observer_modules

        fixed_qparams_observer_to_qconfig = {}
        for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items(
        ):
            if observer in fixed_qparams_observer_to_qconfig:
                fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[
                    observer]
            else:
                activation = FixedQParamsFakeQuantize.with_args(
                    observer=observer)

                fixed_qparams_qconfig = QConfig(
                    activation=activation, weight=default_weight_fake_quant)
                fixed_qparams_observer_to_qconfig[
                    observer] = fixed_qparams_qconfig
            self.qconfig_mapping.set_object_type(fixed_qparams_op,
                                                 fixed_qparams_qconfig)

        self.backend_config = BackendConfigs[self.backend]
        self.example_inputs = (torch.randn(1, 3, 224, 224), )

        self.extra_redundant_fakequants = extra_redundant_fakequants

    @property
    def backend(self):
        """tmp."""
        return 'native'

    @property
    def support_w_modes(self):
        """tmp."""
        return ['per_tensor', 'per_channel']

    @property
    def support_a_modes(self):
        """tmp."""
        return ['per_tensor']

    def prepare(self, model, graph_module):
        """tmp."""
        self.sync_module_training_mode(graph_module, self.is_qat)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)
        prepared = self.del_redundant_fakequant(prepared)

        return prepared

    def post_process_weight_fakequant(self,
                                      observed_module,
                                      keep_fake_quant=False):
        """tmp."""

        def traverse(module):
            for name, child in module.named_children():
                if isinstance(child, SUPPORT_QAT_MODULES):
                    weight_fakequant = child.weight_fake_quant
                    child.weight.data = weight_fakequant(child.weight.data)

                    float_child = child.to_float()

                    if keep_fake_quant:
                        for m in float_child.modules():
                            setattr(m, 'qconfig', self.qconfig.convert())

                        if type(child) in MERGE_BN_MAPPINGS:
                            cls = MERGE_BN_MAPPINGS[type(child)]
                            new_child = cls.from_float(float_child)
                        else:
                            new_child = type(child).from_float(float_child)

                        new_child.weight_fake_quant(new_child.weight)
                    else:
                        new_child = float_child
                    setattr(module, name, new_child)
                else:
                    traverse(child)

        observed_module.apply(enable_fake_quant)
        traverse(observed_module)

    def prepare_for_mmdeploy(self, model, dummy_input, checkpoint):
        """tmp."""
        raise NotImplementedError

    def del_redundant_fakequant(self, prepared):
        """tmp."""
        extra_module_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_module_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_module(
            prepared,
            self.module_prev_wo_fakequant + extra_module_prev_wo_fakequant,
            inplace=True)

        extra_module_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_module_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_module(
            prepared,
            self.module_next_wo_fakequant + extra_module_next_wo_fakequant,
            inplace=True)

        extra_function_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_function_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_method(
            prepared,
            self.function_prev_wo_fakequant + extra_function_prev_wo_fakequant,
            inplace=True)

        extra_function_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_function_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_method(
            prepared,
            self.function_next_wo_fakequant + extra_function_next_wo_fakequant,
            inplace=True)

        extra_method_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_method_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_function(
            prepared,
            self.method_prev_wo_fakequant + extra_method_prev_wo_fakequant,
            inplace=True)

        extra_method_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_method_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_function(
            prepared,
            self.method_next_wo_fakequant + extra_method_next_wo_fakequant,
            inplace=True)

        extra_op_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_op_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_op(
            prepared,
            self.op_prev_wo_fakequant + extra_op_prev_wo_fakequant,
            inplace=True)

        extra_op_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_op_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_op(
            prepared,
            self.op_next_wo_fakequant + extra_op_next_wo_fakequant,
            inplace=True)
        return prepared

    @property
    def module_prev_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def module_next_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def function_prev_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def function_next_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def method_prev_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def method_next_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def op_prev_wo_fakequant(self):
        """tmp."""
        return tuple()

    @property
    def op_next_wo_fakequant(self):
        """tmp."""
        return tuple()
