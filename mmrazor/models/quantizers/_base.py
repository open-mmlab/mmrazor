from abc import abstractmethod
import torch
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.structures.quantization import QConfigHander
from mmengine.model import BaseModule
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str
from torch.nn.intrinsic.qat import modules as qat_fused_modules
from torch.nn.qat import modules as qat_modules

GLOBAL_DICT_KEY = "global"
OBJECT_TYPE_DICT_KEY = "object_type"
MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
MODULE_NAME_DICT_KEY = "module_name"
MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"

FLOAT_TO_OBSERVED_DICT_KEY = "float_to_observed_custom_module_class"
PRESERVED_ATTRIBUTES_DICT_KEY = "preserved_attributes"

class BaseQuantizer(BaseModule):
    def __init__(self,
                 tracer,
                 init_cfg=None):
        super().__init__(init_cfg):
        self.tracer = TASK_UTILS.build(tracer)
    
    @abstractmethod
    def prepare(self):
        pass

@MODELS.register_module()
class WithoutDeployQuantizer(BaseModule):
    def __init__(self,
                 tracer,
                 qconfig_mapping,
                 prepare_custom_config=None,
                 init_cfg=None):
        super().__init__(tracer, init_cfg):
        self.qconfig_mapping = gen_qconfig_mapping(qconfig_mapping)
        self.prepare_custom_config = gen_prepare_custom_config(prepare_custom_config)

    def prepare(self, model, graph_module):
        _swap_ff_with_fxff(model)

        preserved_attributes = self.prepare_custom_config.preserved_attributes
        for attr_name in preserved_attributes:
            setattr(graph_module, attr_name, getattr(model, attr_name))
        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(preserved_attributes)
        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            fuse_custom_config=fuse_custom_config
        )
        prepared = prepare(
            graph_module=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            prepare_custom_config=self.prepare_custom_config
        )
        for attr_name in preserved_attributes:
            setattr(prepared, attr_name, getattr(model, attr_name))
        return prepared

    def gen_qconfig_mapping(self, qconfig_mapping):
        conf = QConfigMapping()
        if GLOBAL_DICT_KEY in qconfig_mapping:
            qconfig = QConfigHander(qconfig_mapping[GLOBAL_DICT_KEY]).convert()
            feed_val = {"": qconfig}
            conf.set_global(feed_val)
        for object_type, qconfig in qconfig_mapping.get(OBJECT_TYPE_DICT_KEY, []):
            qconfig = QConfigHander(qconfig).convert()
            conf.set_object_type(object_type, qconfig)

        for module_name_regex, qconfig in qconfig_mapping.get(MODULE_NAME_REGEX_DICT_KEY, []):
            qconfig = QConfigHander(qconfig).convert()
            conf.set_module_name_regex(module_name_regex, qconfig)
        for module_name, qconfig in qconfig_mapping.get(MODULE_NAME_DICT_KEY, []):
            qconfig = QConfigHander(qconfig).convert()
            conf.set_module_name(module_name, qconfig)
        for module_name, object_type, index, qconfig in qconfig_mapping.get(MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY, []):
            qconfig = QConfigHander(qconfig).convert()
            conf.set_module_name_object_type_order(module_name, object_type, index, qconfig)

        return conf
    
    def gen_prepare_custom_config(self, prepare_custom_config):
        if prepare_custom_config is None:
            prepare_custom_config = None
        else:
            conf = PrepareCustomConfig()
            for quant_type_name, custom_module_mapping in prepare_custom_config.get(FLOAT_TO_OBSERVED_DICT_KEY, {}).items():
                quant_type = _quant_type_from_str(quant_type_name)
                for float_class_str, observed_class_str in custom_module_mapping.items():
                    float_class = MODELS.get(float_class_str)
                    observed_class = MODELS.get(observed_class_str)
                    conf.set_float_to_observed_mapping(float_class, observed_class, quant_type)
            conf.set_preserved_attributes(prepare_custom_config.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return prepare_custom_config

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
class WithDeployQuantizer(BaseModule):

    example_inputs: (torch.randn(1, 3, 224, 224),)
    backend: 'native'

    def __init__(self,
                 global_qconfig,
                 tracer,
                 init_cfg=None):
        super().__init__(tracer, init_cfg):
        qconfig = QConfigHander(global_qconfig).convert()
        self.qconfig_mapping = QConfigMapping().set_global(qconfig)
        self.backend_config = self.get_backend_config()
    
    def prepare(self, model, graph_module):
        _swap_ff_with_fxff(model)
        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config
        )
        prepared = prepare(
            graph_module=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config
        )
        return prepared
    
    def get_backend_config(self):
        return backend_config_mapping[self.backend]
    
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
        pass
    

