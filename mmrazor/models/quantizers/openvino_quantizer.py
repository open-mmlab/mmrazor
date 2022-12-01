# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization import disable_observer
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import DefaultQconfigs
from mmrazor.models.task_modules.tracer.fx.custom_tracer import build_graphmodule
from .base import CustomQuantizer


@MODELS.register_module()
class OpenvinoQuantizer(CustomQuantizer):
    """Quantizer for Openvino backend."""

    support_bits = [8]
    support_w_mode = ['per_channel']
    support_a_mode = ['per_tensor']


    def __init__(self,
                 qconfig,
                 is_qat=True,
                 skipped_methods=None,
                 prepare_custom_config_dict=None,
                 convert_custom_config_dict=None,
                 equalization_qconfig_dict=None,
                 _remove_qconfig=True,
                 init_cfg=None):
        super().__init__(qconfig, is_qat, skipped_methods,
                         prepare_custom_config_dict,
                         convert_custom_config_dict, equalization_qconfig_dict,
                         _remove_qconfig, init_cfg)

        


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




    
    
