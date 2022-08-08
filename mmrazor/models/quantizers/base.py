import torch
from mmengine.model import BaseModel
from mmrazor.structures.quantization import DefalutQconfigs, QuantizeScheme
from mmrazor.structures.quantization import SupportQtypes, CheckArgs
from mmrazor.models.utils import CustomTracer
from torch.ao.quantization.fx import prepare
from torch.ao.quantization import _fuse_fx, _convert_fx, QConfig
from mmrazor.models import BaseObserver, BaseFakeQuant
from mmrazor.registry import MODELS

@MODELS.register_module()
class BaseQuantizer(BaseModel)

    def __init__(self,
                 example_inputs,
                 qconfig=DefalutQconfigs.default, 
                 prepare_custom_config=None,
                 convert_custom_config=None,
                 _remove_qconfig=True,
                 _equalization_config=None,
                 is_standalone_module=False,
                 is_qat=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        # TODO: check_qconfig, qconfig_convert
        if check_qconfig(qconfig):
            self.qconfig = self.qconfig_convert(qconfig)
        else:
            raise
        if prepare_custom_config is None:
            self.prepare_custom_config = PrepareCustomConfig()
        elif isinstance(prepare_custom_config, Dict):
            warnings.warn(
                "Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported "
                "in a future version. Please pass in a PrepareCustomConfig instead.")
            self.prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)
        else:
            self.prepare_custom_config = prepare_custom_config
        
        if _equalization_config is None:
            self._equalization_config = QConfigMapping()
        else:
            self._equalization_config = _equalization_config
        
        self.is_standalone_module = is_standalone_module
        self.is_qat = is_qat
        self.example_inputs = (torch.randn(example_inputs))
        self.convert_custom_config = convert_custom_config
        self._remove_qconfig = _remove_qconfig

    def check_qconfig(self, qconfig):
        is_pass = True
        for arg in CheckArgs:
            if arg == 'qtype':
                if arg in SupportQtypes and arg in qconfig.keys():
                    continue
                else:
                    is_pass = False
                    break
            else:
                if isinstance(arg, dict) and arg in qconfig.keys():
                    continue
                else:
                    is_pass = False
                    break
        return is_pass

    def qconfig_convert(self, qconfig):
        self.w_observer = MODELS.build(qconfig.w_observer)
        self.a_observer = MODELS.build(qconfig.a_observer)
        self.w_fake_quant = MODELS.build(qconfig.w_fake_quant)
        self.a_fake_quant = MODELS.build(qconfig.a_fake_quant)
        self.w_qscheme = QuantizeScheme(**qconfig.w_qscheme)
        self.a_qscheme = QuantizeScheme(**qconfig.a_qscheme)
        w_qconfig = self.w_fake_quant.with_args(observer=self.w_observer, **self.w_qscheme)
        a_qconfig = self.a_fake_quant.with_args(observer=self.a_observer, **self.a_qscheme)
        torch_qconfig = QConfig(weight=w_qconfig, activation=a_qconfig)
        return torch_qconfig

    @abstract
    def preprare(self, model):
        # swap FloatFunctional with FXFloatFunctional
        _swap_ff_with_fxff(model)

        graph_module, tracer = self.trace_model(model)

        graph_module = self.fuse_model(graph_module)
        
        prepared = prepare(
            graph_module,
            self.qconfig,
            self.is_qat,
            tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            prepare_custom_config=self.prepare_custom_config,
            _equalization_config=self._equalization_config,
            is_standalone_module=self.is_standalone_module,
        )  # type: ignore[operator]

        for attr_name in self.prepare_custom_config.preserved_attributes:
            setattr(prepared, attr_name, getattr(model, attr_name))
        return prepared

    @abstract
    def convert(self, graph_module):
        quantized = _convert_fx(
            graph_module,
            is_reference=False,
            convert_custom_config=self.convert_custom_config,
            _remove_qconfig=self._remove_qconfig,
            qconfig_mapping=self.qconfig)
        return quantized

    @static
    def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
        r""" Swap FloatFunctional with FXFloatFunctional
        """
        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(module, torch.nn.quantized.FloatFunctional):
                modules_to_swap.append(name)
            else:
                _swap_ff_with_fxff(module)

        for name in modules_to_swap:
            del model._modules[name]
            model._modules[name] = torch.nn.quantized.FXFloatFunctional()

    def fuse_model(self, graph_module):
        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(self.prepare_custom_config.preserved_attributes)
        graph_module = _fuse_fx(
            graph_module,
            self.is_qat,
            fuse_custom_config)
        return graph_module

    def trace_model(self, model):
        skipped_module_names, skipped_module_classes = \
            get_skipped_module_name_and_classes(self.prepare_custom_config, self.is_standalone_module)
        preserved_attributes = self.prepare_custom_config.preserved_attributes
        # symbolically trace the model
        tracer = CustomTracer(skipped_module_names, skipped_module_classes)  # type: ignore[arg-type]
        graph_module = GraphModule(model, tracer.trace(model))
        for attr_name in preserved_attributes:
            setattr(graph_module, attr_name, getattr(model, attr_name))
        return graph_module, tracer

    