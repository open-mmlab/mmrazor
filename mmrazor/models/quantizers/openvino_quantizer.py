# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.ao.quantization import disable_observer
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.quantize_fx import _fuse_fx

from mmrazor.models.task_modules.tracer.fx import build_graphmodule
from mmrazor.registry import MODELS
from .native_quantizer import NativeQuantizer

MODULE_DEL_PREV_FAKEQUANT = (torch.nn.ReLU6, torch.nn.Identity)
MODULE_DEL_NEXT_FAKEQUANT = (torch.nn.MaxPool2d, )
TARGET_DEL_PREV_FAKEQUANT: Tuple = tuple()
TARGET_DEL_NEXT_FAKEQUANT = ('flatten', )
OP_DEL_PREV_FAKEQUANT = ('output', )
OP_DEL_NEXT_FAKEQUANT: Tuple = tuple()


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

    def prepare(self, model, graph_module):
        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)

        prepared = self.del_fakequant(prepared)

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
            observed_model.load_state_dict(
                torch.load(checkpoint)['state_dict'])
        self.post_process_weight_fakequant(
            observed_model, keep_fake_quant=True)

        observed_model.apply(disable_observer)

        return observed_model

    @property
    def module_del_prev_fakequant(self):
        return (torch.nn.ReLU6, torch.nn.Identity)

    @property
    def module_del_next_fakequant(self):
        return (torch.nn.MaxPool2d, )

    @property
    def method_del_next_fakequant(self):
        return ('flatten', )

    @property
    def op_del_prev_fakequant(self):
        return ('output', )
