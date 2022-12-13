from abc import abstractmethod
import copy
import torch
import torch.nn as nn
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.structures.quantization import QConfigHander, BackendConfigs
from mmrazor.models.utils import str2class
from mmengine.model import BaseModule
from torch.ao.quantization import enable_fake_quant
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.quantize_fx import _fuse_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig, FuseCustomConfig
from torch.nn.intrinsic.qat import modules as qat_fused_modules
from torch.nn.qat import modules as qat_modules
from torch.ao.quantization import FakeQuantizeBase

GLOBAL_DICT_KEY = "_global_"
OBJECT_TYPE_DICT_KEY = "object_type"
MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
MODULE_NAME_DICT_KEY = "module_name"
MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"

FLOAT_TO_OBSERVED_DICT_KEY = "float_to_observed_custom_module_class"
PRESERVED_ATTRIBUTES_DICT_KEY = "preserved_attributes"

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

MODULE_DEL_PREV_FAKEQUANT = (nn.ReLU6, nn.Identity)
MODULE_DEL_NEXT_FAKEQUANT = (nn.MaxPool2d, )
TARGET_DEL_PREV_FAKEQUANT = ('output', )
TARGET_DEL_NEXT_FAKEQUANT = ('flatten', )

def _get_attrs(target, attrs):
    attrs = attrs.split('.')
    for att in attrs:
        target = getattr(target, att, None)
    return target


def del_fakequant_before_target(prepared_model, target_patterns, inplace=True):

    def recursive_find_erased_nodes(node):
        """Find FakeQuant before target node recursively.

        Examples:
            head_fc = self.head.fc(activation_post_process_87);  activation_post_process_87 = None
            activation_post_process_88 = self.activation_post_process_88(head_fc);  head_fc = None
            head = self.head
            _get_loss = head._get_loss(activation_post_process_88, data_samples);  head = activation_post_process_88 = data_samples = None
            return _get_loss

        node                       |           node.args
        --------------------
        output                     | (_get_loss, )
        _get_loss                  | (head, activation_post_process_88, data_samples)
        head                       | ()
        activation_post_process_88 | (head_fc, )
        data_samples               | (None, )
        """
        if node is None:
            return
        if isinstance(_get_attrs(prepared_model, node.target), FakeQuantizeBase):
            nodes_to_erase.append(node)
            return
        for prev_node in node.args:
            recursive_find_erased_nodes(prev_node)
        for prev_node in node.kwargs.values():
            recursive_find_erased_nodes(prev_node)
        return

    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.target in target_patterns:
            nodes_to_erase = []
            recursive_find_erased_nodes(node)
            for to_erase in nodes_to_erase:
                to_erase.replace_all_uses_with(to_erase.args[0])
                new_graph.erase_node(to_erase)
                delattr(prepared_model, to_erase.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_target(prepared_model, target_patterns, inplace=True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node.target not in target_patterns:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_before_module(prepared_model, module_patterns, inplace=True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(_get_attrs(prepared_model, node.target), module_patterns):
            to_erase = node.args[0]
            if not isinstance(_get_attrs(prepared_model, to_erase.target), FakeQuantizeBase):
                continue
            if len(to_erase.users) > 1:
                continue
            to_erase.replace_all_uses_with(to_erase.args[0])
            new_graph.erase_node(to_erase)
            delattr(prepared_model, to_erase.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_module(prepared_model, module_patterns, inplace=True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(_get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if not isinstance(_get_attrs(prepared_model, prev_node.target),
                              module_patterns):
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


class BaseQuantizer(BaseModule):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = TASK_UTILS.build(tracer)
    
    @abstractmethod
    def prepare(self):
        pass
    
    def swap_ff_with_fxff(self, model):
        r""" Swap FloatFunctional with FXFloatFunctional
        """
        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
                modules_to_swap.append(name)
            else:
                self.swap_ff_with_fxff(module)

        for name in modules_to_swap:
            del model._modules[name]
            model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()

@MODELS.register_module()
class WithoutDeployQuantizer(BaseQuantizer):
    # TODO: add backendconfig
    def __init__(self,
                 qconfig_mapping,
                 tracer=dict(type='mmrazor.CustomTracer'),
                 prepare_custom_config=None):
        super().__init__(tracer)
        self.qconfig_mapping = self.gen_qconfig_mapping(qconfig_mapping)
        self.prepare_custom_config = self.gen_prepare_custom_config(prepare_custom_config)
        self.example_inputs = (torch.randn(1, 3, 224, 224),)

    def prepare(self, model, graph_module):
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
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            prepare_custom_config=self.prepare_custom_config
        )
        for attr_name in preserved_attributes:
            setattr(prepared, attr_name, getattr(model, attr_name))

        prepared = del_fakequant_before_module(prepared, MODULE_DEL_PREV_FAKEQUANT, inplace=True)
        prepared = del_fakequant_after_module(prepared, MODULE_DEL_NEXT_FAKEQUANT, inplace=True)
        prepared = del_fakequant_before_target(prepared, TARGET_DEL_PREV_FAKEQUANT, inplace=True)
        prepared = del_fakequant_after_target(prepared, TARGET_DEL_NEXT_FAKEQUANT, inplace=True)


        return prepared

    def gen_qconfig_mapping(self, qconfig_mapping):
        conf = QConfigMapping()
        if GLOBAL_DICT_KEY in qconfig_mapping:
            qconfig = QConfigHander(qconfig_mapping[GLOBAL_DICT_KEY]).convert()
            conf.set_global(qconfig)
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
        conf = PrepareCustomConfig()
        if prepare_custom_config is None:
            return conf
        else:
            for quant_type_name, custom_module_mapping in prepare_custom_config.get(FLOAT_TO_OBSERVED_DICT_KEY, {}).items():
                quant_type = _quant_type_from_str(quant_type_name)
                for float_class_str, observed_class_str in custom_module_mapping.items():
                    float_class = MODELS.get(float_class_str)
                    observed_class = MODELS.get(observed_class_str)
                    conf.set_float_to_observed_mapping(float_class, observed_class, quant_type)
            conf.set_preserved_attributes(prepare_custom_config.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
            return conf

@MODELS.register_module()
class WithDeployQuantizer(BaseQuantizer):

    # backend: 'native'
    # support_w_modes = ['per_tensor', 'per_channel']
    # support_a_modes = ['per_tensor']

    def __init__(self,
                 global_qconfig,
                 no_observer_modules=None,
                 tracer=dict(type='CustomTracer')):
        super().__init__(tracer)
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
        self.no_observer_modules = str2class(no_observer_modules)
        self.qconfig_mapping = QConfigMapping().set_global(self.qconfig.convert())
        for mod in self.no_observer_modules:
            import pdb; pdb.set_trace()
            self.qconfig_mapping.set_object_type(mod, None)
        self.backend_config = BackendConfigs[self.backend]
        self.example_inputs = (torch.randn(1, 3, 224, 224),)
    
    @property
    def backend(self):
        return 'native'
    
    @property
    def support_w_modes(self):
        return ['per_tensor', 'per_channel']

    @property
    def support_a_modes(self):
        return ['per_tensor']
    
    def prepare(self, model, graph_module):
        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config
        )
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config
        )

        prepared = del_fakequant_before_module(prepared,
                                               MODULE_DEL_PREV_FAKEQUANT,
                                               inplace=True)
        prepared = del_fakequant_after_module(prepared,
                                              MODULE_DEL_NEXT_FAKEQUANT,
                                              inplace=True)
        prepared = del_fakequant_before_target(prepared,
                                               TARGET_DEL_PREV_FAKEQUANT,
                                               inplace=True)
        prepared = del_fakequant_after_target(prepared,
                                              TARGET_DEL_NEXT_FAKEQUANT,
                                              inplace=True)

        return prepared
    
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
        raise NotImplementedError
