import torch
from torch.fx import GraphModule
from mmengine.model import BaseModule
from mmrazor.structures.quantization import DefalutQconfigs, QuantizeScheme
from mmrazor.structures.quantization import SupportQtypes, CheckArgs
from mmrazor.models.utils import CustomTracer
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.quantize_fx import _fuse_fx, _convert_fx, QuantizationTracer
from torch.ao.quantization import QConfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, prepare_fx, convert_fx
from mmrazor.registry import MODELS
from mmrazor.models.utils import (
    check_is_valid_convert_custom_config_dict,
    check_is_valid_fuse_custom_config_dict,
    check_is_valid_prepare_custom_config_dict,
    check_is_valid_qconfig_dict,
    get_custom_module_class_keys
)

@MODELS.register_module()
class CustomQuantizer(BaseModule):

    def __init__(self,
                 qconfig=DefalutQconfigs['default'],
                 is_qat=True,
                 prepare_custom_config_dict=None,
                 convert_custom_config_dict=None,
                 equalization_qconfig_dict=None,
                 _remove_qconfig=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        if self.check_qconfig(qconfig):
            qconfig = self.qconfig_convert(qconfig)
            self.qconfig_dict = {"": qconfig}
        else:
            raise ValueError('qconfig is uncorrect!')
        
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
        check_is_valid_prepare_custom_config_dict(self.prepare_custom_config_dict)
        check_is_valid_convert_custom_config_dict(self.convert_custom_config_dict)
        check_is_valid_qconfig_dict(self.equalization_qconfig_dict)
        
        self.is_qat = is_qat
        self._remove_qconfig = _remove_qconfig

    def prepare(self, model):
        # swap FloatFunctional with FXFloatFunctional
        self._swap_ff_with_fxff(model)

        tracer = self.build_tracer()
        graph_module = GraphModule(model, tracer.trace(model, concrete_args={'mode': 'tensor'}))

        preserved_attributes = self.prepare_custom_config_dict.get("preserved_attributes", [])
        for attr_name in preserved_attributes:
            setattr(graph_module, attr_name, getattr(model, attr_name))
        
        graph_module = self.fuse_model(graph_module)
        
        prepared = prepare(
            graph_module,
            self.qconfig_dict,
            self.is_qat,
            tracer.node_name_to_scope,
            prepare_custom_config_dict=self.prepare_custom_config_dict,
            equalization_qconfig_dict=self.equalization_qconfig_dict
        )  # type: ignore[operator]

        for attr_name in preserved_attributes:
            setattr(prepared, attr_name, getattr(model, attr_name))
        return prepared

    def convert(self, graph_module):
        quantized = _convert_fx(
            graph_module,
            is_reference=False,
            convert_custom_config=self.convert_custom_config,
            _remove_qconfig=self._remove_qconfig,
            qconfig_mapping=self.qconfig)
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
        
        w_observer = MODELS.build(qconfig['w_observer'])
        a_observer = MODELS.build(qconfig['a_observer'])
        self.w_observer = w_observer.with_args(**self.w_qscheme.to_observer_params())
        self.a_observer = a_observer.with_args(**self.a_qscheme.to_observer_params())

        # w_fake_quant = MODELS.build(qconfig['w_fake_quant'])
        # a_fake_quant = MODELS.build(qconfig['a_fake_quant'])
        # self.w_fake_quant = w_fake_quant.with_args(observer=self.w_observer)
        # self.a_fake_quant = a_fake_quant.with_args(observer=self.a_observer)
        self.w_fake_quant = MODELS.build(qconfig['w_fake_quant'])
        self.a_fake_quant = MODELS.build(qconfig['a_fake_quant'])
        
        torch_qconfig = QConfig(weight=self.w_fake_quant, activation=self.a_fake_quant)
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
            "non_traceable_module_name", []
        )
        skipped_module_classes = self.prepare_custom_config_dict.get(
            "non_traceable_module_class", []
        )  
        standalone_module_name_configs = self.prepare_custom_config_dict.get(
            "standalone_module_name", []
        )
        skipped_module_names += [config[0] for config in standalone_module_name_configs]

        standalone_module_class_configs = self.prepare_custom_config_dict.get(
            "standalone_module_class", []
        )
        skipped_module_classes += [
            config[0] for config in standalone_module_class_configs
        ]
        float_custom_module_classes = get_custom_module_class_keys(
            self.prepare_custom_config_dict, "float_to_observed_custom_module_class"
        )
        skipped_module_classes += float_custom_module_classes
        # tracer = CustomTracer(skipped_module_names, skipped_module_classes)
        tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
        return tracer
    
    def fuse_model(self, graph_module):
        graph_module = _fuse_fx(
            graph_module,
            self.is_qat,
            self.prepare_custom_config_dict)
        return graph_module


# @MODELS.register_module()
# class BaseQuantizer(BaseModule):

#     def __init__(self,
#                  qconfig=DefalutQconfigs['default'], 
#                  prepare_custom_config=None,
#                  convert_custom_config=None,
#                  _remove_qconfig=True,
#                  equalization_qconfig=None,
#                  is_qat=True,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         if self.check_qconfig(qconfig):
#             self.qconfig = self.qconfig_convert(qconfig)
#         else:
#             raise ValueError('qconfig is uncorrect!')
#         self.prepare_custom_config = prepare_custom_config    
#         self.equalization_qconfig = equalization_qconfig
#         self.is_qat = is_qat
#         self.convert_custom_config = convert_custom_config
#         self._remove_qconfig = _remove_qconfig

#     def check_qconfig(self, qconfig):
#         is_pass = True
#         for arg in CheckArgs:
#             if arg == 'qtype':
#                 if qconfig[arg] in SupportQtypes and arg in qconfig.keys():
#                     continue
#                 else:
#                     is_pass = False
#                     break
#             else:
#                 if isinstance(qconfig[arg], dict) and arg in qconfig.keys():
#                     continue
#                 else:
#                     is_pass = False
#                     break
#         return is_pass

#     def qconfig_convert(self, qconfig):
#         self.w_qscheme = QuantizeScheme(**qconfig['w_qscheme'])
#         self.a_qscheme = QuantizeScheme(**qconfig['a_qscheme'])
        
#         w_observer = MODELS.build(qconfig['w_observer'])
#         a_observer = MODELS.build(qconfig['a_observer'])
#         self.w_observer = w_observer.with_args(**self.w_qscheme.to_observer_params())
#         self.a_observer = a_observer.with_args(**self.a_qscheme.to_observer_params())

#         w_fake_quant = MODELS.build(qconfig['w_fake_quant'])
#         a_fake_quant = MODELS.build(qconfig['a_fake_quant'])
#         self.w_fake_quant = w_fake_quant.with_args(observer=self.w_observer)
#         self.a_fake_quant = a_fake_quant.with_args(observer=self.a_observer)
        
#         torch_qconfig = QConfig(weight=self.w_fake_quant, activation=self.a_fake_quant)
#         return torch_qconfig

#     def prepare(self, model):
#         qconfig_dict = {"": self.qconfig}
#         if self.is_qat:
#             prepared_model = prepare_qat_fx(
#                 model=model,
#                 qconfig_dict=qconfig_dict,
#                 prepare_custom_config_dict=self.prepare_custom_config
#             )
#         else:
#             prepared_model = prepare_fx(
#                 model=model,
#                 qconfig_dict=qconfig_dict,
#                 prepare_custom_config_dict=self.prepare_custom_config,
#                 equalization_qconfig_dict=self.equalization_qconfig
#             )
#         return prepared_model

#     def convert(self, graph_module):
#         return convert_fx(
#             graph_module=graph_module,
#             is_reference=self.is_reference,
#             convert_custom_config_dict=self.convert_custom_config,
#             _remove_qconfig=self._remove_qconfig,
#             qconfig_dict={"": self.qconfig}
#         )