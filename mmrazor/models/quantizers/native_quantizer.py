import torch
import torch.nn as nn
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import QConfigHander, BackendConfigs
from mmrazor.models.utils import str2class
from torch.ao.quantization import enable_fake_quant
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.quantize_fx import _fuse_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.nn.intrinsic.qat import modules as qat_fused_modules
from torch.nn.qat import modules as qat_modules

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
class NativeQuantizer(BaseQuantizer):

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
