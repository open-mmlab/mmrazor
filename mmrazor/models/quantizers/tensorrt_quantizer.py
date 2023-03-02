# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch

try:
    from torch.ao.quantization import disable_observer
except ImportError:
    from mmrazor.utils import get_placeholder
    disable_observer = get_placeholder('torch>=1.13')

from mmrazor.registry import MODELS
from mmrazor.structures.quantization import QConfigHandler
from .native_quantizer import NativeQuantizer


@MODELS.register_module()
class TensorRTQuantizer(NativeQuantizer):
    """Quantizer for quantizing and deploying to TensorRT backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    TensorRT's some important features about quantization is as follows:
    * support_w_mode = ('per_tensor', 'per_channel')
    * support_a_mode = ('per_tensor')
    """

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'tensorrt'

    @property
    def support_w_modes(self):
        """Supported quantization modes for weight about per_tensor or
        per_channel."""
        return ('per_tensor', 'per_channel')

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor')

    def prepare_for_mmdeploy(self,
                             model: torch.nn.Module,
                            #  dummy_input: Tuple = (1, 3, 224, 224),
                             dummy_input=None,
                             checkpoint: Optional[str] = None):
        """Prepare for deploy to the backend with mmdeploy, which will be used
        in mmdeploy, and usually includes as follows:

        1. prepare for the float model rewritten by mmdeploy.
        2. load checkpoint consists of float weight and quantized params in
        mmrazor.
        3. post process weight fakequant for exporting .onnx that meet
        the backend's requirement.
        """
        observed_model = self.prepare(model)
        if dummy_input is not None:
            observed_model(torch.randn(dummy_input).cuda())
        if checkpoint is not None:
            observed_model.load_state_dict(
                torch.load(checkpoint)['state_dict'])
        
        self.post_process_weight_fakequant(
            observed_model, 
            device='cuda',
            keep_fake_quant=True)
        
        for node in observed_model.graph.nodes:
            if 'activation_post_process_' in node.name:
                module_name = node.target
                module = getattr(observed_model, module_name)
                fakequant_new = QConfigHandler.replace_fakequant(module, self.qconfig.a_qscheme, update_qparams=True)
                setattr(observed_model, module_name, fakequant_new)
        
        observed_model.apply(disable_observer)
        
        return observed_model
    
    @property
    def module_prev_wo_fakequant(self):
        """Configurate the modules that their previous nodes are redundant
        fakequants."""
        return (torch.nn.ReLU6, torch.nn.Identity)

    @property
    def module_next_wo_fakequant(self):
        """Configurate the modules that their next nodes are redundant
        fakequants."""
        return (torch.nn.MaxPool2d, )

    @property
    def method_next_wo_fakequant(self):
        """Configurate the methods that their next nodes are redundant
        fakequants."""
        return ('flatten', )

    @property
    def op_prev_wo_fakequant(self):
        """Configurate the OPs that their previous nodes are redundant
        fakequants."""
        return ('output', )
