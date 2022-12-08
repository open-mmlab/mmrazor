# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization import disable_observer
from mmrazor.registry import MODELS
from mmrazor.models.task_modules.tracer.fx.custom_tracer import build_graphmodule
from .base import WithDeployQuantizer


@MODELS.register_module()
class OpenVINOQuantizer(WithDeployQuantizer):
    """Quantizer for Openvino backend."""

    backend: 'openvino'
    support_w_mode = ['per_tensor', 'per_channel']
    support_a_mode = ['per_tensor']

    def __init__(self,
                 global_qconfig,
                 tracer=dict(type='CustomTracer'),
                 init_cfg=None):
        super().__init__(global_qconfig, tracer, init_cfg)

    def prepare_for_mmdeploy(self, model, dummy_input=None, checkpoint=None):
        

        self._swap_ff_with_fxff(model)
        graph = self.tracer.trace(model)
        graph_module = build_graphmodule(model, graph)
        observed_model = self.prepare(model, graph_module)
        
        
        self.post_process_weight_fakequant(observed_model, keep_fake_quant=True)
        
        if dummy_input is not None:
            observed_model(dummy_input)

        observed_model.apply(disable_observer)

        return observed_model




    
    
