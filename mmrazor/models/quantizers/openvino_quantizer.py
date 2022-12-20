# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization import disable_observer
from mmrazor.registry import MODELS
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.quantize_fx import _fuse_fx
from mmrazor.models.task_modules.tracer.fx.custom_tracer import build_graphmodule
from .base import NativeQuantizer


@MODELS.register_module()
class OpenVINOQuantizer(NativeQuantizer):
    """Quantizer for Openvino backend."""

    # backend: 'openvino'
    # support_w_mode = ['per_tensor', 'per_channel']
    # support_a_mode = ['per_tensor']

    def __init__(self,
                 global_qconfig,
                 no_observer_modules=None,
                 tracer=dict(type='CustomTracer'),
                 remove_fakequants=dict(
                    module_prev=[nn.ReLU6, nn.Identity],
                    module_next=[nn.MaxPool2d],
                    target_prev=['output'],
                    target_next=['flatten']
                 )):
        super().__init__(global_qconfig, no_observer_modules, tracer)
        self.remove_fakequants = remove_fakequants

    @property
    def backend(self):
        return 'openvino'

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
        module_prev = self.remove_fakequants.get('module_prev', [])
        module_next = self.remove_fakequants.get('module_next', [])
        target_prev = self.remove_fakequants.get('target_prev', [])
        target_next = self.remove_fakequants.get('target_next', [])

        prepared = del_fakequant_before_module(prepared,
                                               module_prev,
                                               inplace=True)
        prepared = del_fakequant_after_module(prepared,
                                              module_next,
                                              inplace=True)
        prepared = del_fakequant_before_target(prepared,
                                               target_prev,
                                               inplace=True)
        prepared = del_fakequant_after_target(prepared,
                                              target_next,
                                              inplace=True)

        return prepared
    
    def prepare_for_mmdeploy(self, 
                             model, 
                             dummy_input=(1, 3, 224, 224),
                             checkpoint=None):

        self.swap_ff_with_fxff(model)
        graph = self.tracer.trace(model)
        graph_module = build_graphmodule(model, graph)
        observed_model = self.prepare(model, graph_module)
        if dummy_input is not None:
            observed_model(torch.randn(dummy_input))
        if checkpoint is not None:
            observed_model.load_state_dict(torch.load(checkpoint)['state_dict'])
        self.post_process_weight_fakequant(
            observed_model, keep_fake_quant=True)

        observed_model.apply(disable_observer)

        return observed_model
