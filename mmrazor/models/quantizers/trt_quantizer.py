# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization import disable_observer

from mmrazor.models.task_modules.tracer.fx.custom_tracer import \
    build_graphmodule
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import DefaultQconfigs
from .base import CustomQuantizer


@MODELS.register_module()
class TensorRTQuantizer(CustomQuantizer):
    """Quantizer for TensorRT backend."""

    support_bits = [8]
    support_w_mode = ['per_channel']
    support_a_mode = ['per_tensor']

    def __init__(self,
                 qconfig=DefaultQconfigs['tensorrt'],
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

        graph = self.tracer.trace(model)
        graph_module = build_graphmodule(model, graph)
        observed_model = self.prepare(model, graph_module)

        observed_model(torch.randn(1, 3, 224, 224))

        self.post_process_weight_fakequant(observed_model)
        if dummy_input is not None:
            observed_model(dummy_input)

        observed_model.apply(disable_observer)

        return observed_model
