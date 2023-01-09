# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization import disable_observer

from mmrazor.models.task_modules.tracer.fx import build_graphmodule
from mmrazor.registry import MODELS
from .native_quantizer import NativeQuantizer


@MODELS.register_module()
class OpenVINOQuantizer(NativeQuantizer):
    """Quantizer for Openvino backend."""

    # backend: 'openvino'
    # support_w_mode = ['per_tensor', 'per_channel']
    # support_a_mode = ['per_tensor']

    @property
    def backend(self):
        return 'openvino'

    @property
    def support_w_modes(self):
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
            observed_model.load_state_dict(
                torch.load(checkpoint)['state_dict'])
        self.post_process_weight_fakequant(
            observed_model, keep_fake_quant=True)

        observed_model.apply(disable_observer)

        return observed_model

    @property
    def module_prev_wo_fakequant(self):
        return (torch.nn.ReLU6, torch.nn.Identity)

    @property
    def module_next_wo_fakequant(self):
        return (torch.nn.MaxPool2d, )

    @property
    def method_next_wo_fakequant(self):
        return ('flatten', )

    @property
    def op_prev_wo_fakequant(self):
        return ('output', )
