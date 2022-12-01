# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmengine.model import BaseModule
from torch.ao.quantization import QConfig, enable_fake_quant
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.quantize_fx import _convert_fx, _fuse_fx
from torch.nn.intrinsic.qat import modules as qat_fused_modules
from torch.nn.qat import modules as qat_modules

from mmrazor.models.task_modules.tracer import CustomTracer
from mmrazor.models.utils import (check_is_valid_convert_custom_config_dict,
                                  check_is_valid_prepare_custom_config_dict,
                                  check_is_valid_qconfig_dict,
                                  get_custom_module_class_keys)
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import (CheckArgs, DefaultQconfigs,
                                             QuantizeScheme, SupportQtypes)

SUPPORT_QAT_MODULES = (
    qat_fused_modules.ConvBn1d, qat_fused_modules.ConvBn2d,
    qat_fused_modules.ConvBn3d, qat_fused_modules.ConvBnReLU1d,
    qat_fused_modules.ConvBnReLU2d, qat_fused_modules.ConvBnReLU3d,
    qat_fused_modules.ConvReLU1d, qat_fused_modules.ConvReLU2d,
    qat_fused_modules.ConvReLU3d, qat_fused_modules.LinearBn1d,
    qat_fused_modules.LinearReLU, qat_modules.Conv1d, qat_modules.Conv2d,
    qat_modules.Conv3d, qat_modules.Linear)

MERGE_BN_MAPPINGS = {
    qat_fused_modules.ConvBn1d: qat_modules.Conv1d,
    qat_fused_modules.ConvBn2d: qat_modules.Conv2d,
    qat_fused_modules.ConvBn3d: qat_modules.Conv3d,
    qat_fused_modules.ConvBnReLU1d: qat_fused_modules.ConvReLU1d,
    qat_fused_modules.ConvBnReLU2d: qat_fused_modules.ConvReLU2d,
    qat_fused_modules.ConvBnReLU3d: qat_fused_modules.ConvReLU3d,
    qat_fused_modules.LinearBn1d: qat_modules.Linear
}


@MODELS.register_module()
class CustomQuantizer(BaseModule):
    """Configurable quantizer, base class of quantizers.

    Args:
        qconfig (Dict, optional): QConfig. Defaults to DefaultQconfigs['default'].  # noqa: E501
        skipped_methods (List, optional): Skipped methods list for tracer.
            Defaults to None.
        prepare_custom_config_dict (Dict, optional): `PrepareCustomConfig`
            from `torch.quantization.fx`. Defaults to None.
        convert_custom_config_dict (Dict, optional): `ConvertCustomConfig`
            from `torch.quantization.fx`. Defaults to None.
        equalization_qconfig_dict (Dict, optional): Custom `QConfig` effects
            on all modules. Defaults to None.
        _remove_qconfig (Dict, optional): Remove qconfig at the end of
            `_convert_fx`. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 qconfig: Dict = DefaultQconfigs['default'],
                 skipped_methods: List = None,
                 prepare_custom_config_dict: Dict = None,
                 convert_custom_config_dict: Dict = None,
                 equalization_qconfig_dict: Dict = None,
                 _remove_qconfig: bool = True,
                 init_cfg: Dict = None):
        super().__init__(init_cfg)
        if self.check_qconfig(qconfig):
            qconfig = self.qconfig_convert(qconfig)
            self.qconfig_dict = {'': qconfig}
        else:
            raise ValueError('qconfig is incorrect!')

        if prepare_custom_config_dict is None:
            self.prepare_custom_config_dict = {}
        else:
            self.prepare_custom_config_dict = prepare_custom_config_dict
        if convert_custom_config_dict is None:
            self.convert_custom_config_dict = {}
        else:
            self.convert_custom_config_dict = convert_custom_config_dict
        if equalization_qconfig_dict is None:
            self.equalization_qconfig_dict = {}
        else:
            self.equalization_qconfig_dict = equalization_qconfig_dict

        check_is_valid_qconfig_dict(self.qconfig_dict)
        check_is_valid_prepare_custom_config_dict(
            self.prepare_custom_config_dict)
        check_is_valid_convert_custom_config_dict(
            self.convert_custom_config_dict)
        check_is_valid_qconfig_dict(self.equalization_qconfig_dict)

        self.skipped_methods = skipped_methods
        self._remove_qconfig = _remove_qconfig
        self.tracer = self.build_tracer()

    def prepare(self, model, graph_module):

        preserved_attributes = self.prepare_custom_config_dict.get(
            'preserved_attributes', [])
        for attr_name in preserved_attributes:
            setattr(graph_module, attr_name, getattr(model, attr_name))

        graph_module = self.fuse_model(graph_module)

        prepared = prepare(
            graph_module,
            self.qconfig_dict,
            True,
            self.tracer.node_name_to_scope,
            equalization_qconfig_dict=self.equalization_qconfig_dict
        )  # type: ignore[operator]

        for attr_name in preserved_attributes:
            setattr(prepared, attr_name, getattr(model, attr_name))
        return prepared

    def convert(self, graph_module):
        quantized = _convert_fx(
            graph_module,
            is_reference=False,
            convert_custom_config_dict=self.convert_custom_config_dict,
            _remove_qconfig=self._remove_qconfig,
            qconfig_dict=self.qconfig_dict)
        return quantized

    def check_qconfig(self, qconfig):
        is_pass = True
        for arg in CheckArgs:
            if arg == 'qtype':
                if qconfig[arg] in SupportQtypes and arg in qconfig.keys():
                    continue
                else:
                    is_pass = False
                    break
            else:
                if isinstance(qconfig[arg], dict) and arg in qconfig.keys():
                    continue
                else:
                    is_pass = False
                    break
        return is_pass

    def qconfig_convert(self, qconfig):
        self.w_qscheme = QuantizeScheme(**qconfig['w_qscheme'])
        self.a_qscheme = QuantizeScheme(**qconfig['a_qscheme'])
        w_observer = MODELS.get(qconfig['w_observer']['type'])
        w_observer_kwargs = self.w_qscheme.to_observer_params()
        a_observer = MODELS.get(qconfig['a_observer']['type'])
        a_observer_kwargs = self.a_qscheme.to_observer_params()
        self.w_observer = MODELS.get(qconfig['w_observer']['type']).with_args(
            **self.w_qscheme.to_observer_params())
        self.a_observer = MODELS.get(qconfig['a_observer']['type']).with_args(
            **self.a_qscheme.to_observer_params())
        self.w_fake_quant = MODELS.get(
            qconfig['w_fake_quant']['type']).with_args(
                observer=w_observer, **w_observer_kwargs)
        self.a_fake_quant = MODELS.get(
            qconfig['a_fake_quant']['type']).with_args(
                observer=a_observer, **a_observer_kwargs)

        torch_qconfig = QConfig(
            weight=self.w_fake_quant, activation=self.a_fake_quant)
        return torch_qconfig

    def _swap_ff_with_fxff(self, model: torch.nn.Module) -> None:
        r""" Swap FloatFunctional with FXFloatFunctional
        """
        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(module, torch.nn.quantized.FloatFunctional):
                modules_to_swap.append(name)
            else:
                self._swap_ff_with_fxff(module)

        for name in modules_to_swap:
            del model._modules[name]
            model._modules[name] = torch.nn.quantized.FXFloatFunctional()

    def build_tracer(self):
        skipped_module_names = self.prepare_custom_config_dict.get(
            'non_traceable_module_name', [])
        skipped_module_classes = self.prepare_custom_config_dict.get(
            'non_traceable_module_class', [])
        standalone_module_name_configs = self.prepare_custom_config_dict.get(
            'standalone_module_name', [])
        skipped_module_names += [
            config[0] for config in standalone_module_name_configs
        ]

        standalone_module_class_configs = self.prepare_custom_config_dict.get(
            'standalone_module_class', [])
        skipped_module_classes += [
            config[0] for config in standalone_module_class_configs
        ]
        float_custom_module_classes = get_custom_module_class_keys(
            self.prepare_custom_config_dict,
            'float_to_observed_custom_module_class')
        skipped_module_classes += float_custom_module_classes
        tracer = CustomTracer(self.skipped_methods, skipped_module_names,
                              skipped_module_classes)
        # tracer = QuantizationTracer(skipped_module_names,
        #    skipped_module_classes)
        return tracer

    def fuse_model(self, graph_module):
        graph_module = _fuse_fx(graph_module, True,
                                self.prepare_custom_config_dict)
        return graph_module

    def post_process_weight_fakequant(self,
                                      observed_module,
                                      keep_fake_quant=False):

        def traverse(module):

            for name, child in module.named_children():
                if isinstance(child, SUPPORT_QAT_MODULES):
                    weight_fakequant = child.weight_fake_quant
                    child.weight.data = weight_fakequant(child.weight.data)

                    float_child = child.to_float()

                    if keep_fake_quant:
                        for m in float_child.modules():
                            setattr(m, 'qconfig', self.qconfig_dict[''])

                        if type(child) in MERGE_BN_MAPPINGS:
                            cls = MERGE_BN_MAPPINGS[type(child)]
                            new_child = cls.from_float(float_child)
                        else:
                            new_child = child.from_float(float_child)

                        new_child.weight_fake_quant(new_child.weight)
                    else:
                        new_child = float_child
                    setattr(module, name, new_child)
                else:
                    traverse(child)
        observed_module.apply(enable_fake_quant)
        traverse(observed_module)

    def prepare_for_mmdeploy(self, model):
        raise NotImplementedError
