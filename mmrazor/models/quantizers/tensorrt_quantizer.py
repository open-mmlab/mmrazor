# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization import disable_observer
from mmrazor.models.task_modules.tracer.fx.custom_tracer import \
    build_graphmodule
from mmrazor.registry import MODELS
from .base import NativeQuantizer


@MODELS.register_module()
class TensorRTQuantizer(NativeQuantizer):
    """Quantizer for TensorRT backend."""

    # backend: 'tensorrt'
    # support_w_mode = ['per_tensor', 'per_channel']
    # support_a_mode = ['per_tensor']

    def __init__(self,
                 global_qconfig,
                 no_observer_modules=None,
                 tracer=dict(type='CustomTracer')):
        super().__init__(global_qconfig, no_observer_modules, tracer)
    
    @property
    def backend(self):
        return 'tensorrt'
    
    @property
    def support_a_modes(self):
        return ['per_tensor', 'per_channel']
    
    @property
    def support_a_modes(self):
        return ['per_tensor']

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
